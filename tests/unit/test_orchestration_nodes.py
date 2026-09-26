from enterprise_ai.orchestration.nodes import router_node
from unittest.mock import Mock, patch

from enterprise_ai.sql_agent.models import SQLQueryResult
from enterprise_ai.rag.models import RAGResult

from enterprise_ai.orchestration.verification import contains_causal_claim
import json

class FakeLLM:
    def generate(self, messages, max_new_tokens=200):
        return "sql"


def test_router_node_adds_route_to_state():
    state = {
        "question": "How many customers are there?"
    }

    result = router_node(
        state=state,
        llm=FakeLLM(),
    )

    assert result == {
        "route": "sql",
    }

def test_sql_node_stores_structured_sql_evidence():
    from enterprise_ai.orchestration.nodes import sql_node

    expected_result = SQLQueryResult(
        sql="SELECT COUNT(*) AS customer_count FROM customers",
        rows=[
            {"customer_count": 10000}
        ],
        truncated=False,
    )

    fake_llm = Mock()
    fake_engine = Mock()

    with patch(
        "enterprise_ai.orchestration.nodes.answer_sql_question",
        return_value=expected_result,
    ) as mock_answer:
        result = sql_node(
            state={
                "question": "How many customers are there?"
            },
            llm=fake_llm,
            engine=fake_engine,
        )

    assert result["sql_evidence"] == expected_result

    mock_answer.assert_called_once_with(
        question="How many customers are there?",
        llm=fake_llm,
        engine=fake_engine,
    )

def test_rag_node_stores_structured_rag_evidence():
    from enterprise_ai.orchestration.nodes import rag_node

    expected_result = RAGResult(
        answer="The incident involved payment failures.",
        sources=[],
    )

    fake_llm = Mock()
    fake_embedding_model = Mock()
    fake_reranking_model = Mock()

    with patch(
        "enterprise_ai.orchestration.nodes.answer_question",
        return_value=expected_result,
    ) as mock_answer:
        result = rag_node(
            state={
                "question": "What happened during the billing incident?"
            },
            llm=fake_llm,
            embedding_model=fake_embedding_model,
            reranking_model=fake_reranking_model,
        )

    assert result["rag_evidence"] == expected_result

    mock_answer.assert_called_once_with(
        question="What happened during the billing incident?",
        llm=fake_llm,
        embedding_model=fake_embedding_model,
        reranking_model=fake_reranking_model,
    )

def test_synthesis_node_combines_sql_and_rag_evidence():
    from enterprise_ai.orchestration.nodes import synthesis_node
    from enterprise_ai.rag.models import RAGResult
    from enterprise_ai.sql_agent.models import SQLQueryResult

    class FakeLLM:
        def generate(self, messages, max_new_tokens=250):
            user_message = messages[1]["content"]

            assert "52.13" in user_message
            assert "billing failures" in user_message.lower()

            return "The evidence shows elevated churn alongside billing problems."

    state = {
        "question": "Why did churn increase?",
        "sql_evidence": SQLQueryResult(
            sql="SELECT 52.13 AS churn_rate",
            rows=[{"churn_rate": 52.13}],
        ),
        "rag_evidence": RAGResult(
            answer="Billing failures affected customers.",
            sources=[],
        ),
    }

    result = synthesis_node(
        state=state,
        llm=FakeLLM(),
    )

    assert result["draft_answer"] == (
        "The evidence shows elevated churn alongside billing problems."
    )

def test_verification_node_revises_unsupported_causal_claim():

    from enterprise_ai.orchestration.nodes import verification_node
    from enterprise_ai.rag.models import RAGResult
    from enterprise_ai.sql_agent.models import SQLQueryResult

    class FakeLLM:
        def generate(self, messages, max_new_tokens=350):
            user_message = messages[1]["content"]

            assert "52.13" in user_message
            assert "billing failures" in user_message.lower()
            assert "caused churn" in user_message.lower()

            return """
            {
                "status": "revise",
                "issues": [
                    "The evidence does not establish that the billing incident caused churn."
                ],
                "final_answer": "Churn was 52.13%, while billing failures were documented during the same period. The available evidence does not establish causation."
            }
            """

    state = {
        "question": "Why did churn increase?",
        "sql_evidence": SQLQueryResult(
            sql="SELECT 52.13 AS churn_rate",
            rows=[{"churn_rate": 52.13}],
        ),
        "rag_evidence": RAGResult(
            answer="Billing failures affected customers.",
            sources=[],
        ),
        "draft_answer": "The billing incident caused churn.",
    }

    result = verification_node(
        state=state,
        llm=FakeLLM(),
    )

    assert "verification" in result
    assert result["final_answer"] == (
        "Churn was 52.13%, while billing failures were documented during "
        "the same period. The available evidence does not establish causation."
    )

def test_verification_node_overrides_incorrect_pass_for_unsupported_causality():
    from enterprise_ai.orchestration.nodes import verification_node
    from enterprise_ai.rag.models import RAGResult, RAGSource
    from enterprise_ai.sql_agent.models import SQLQueryResult

    class FakeLLM:
        def generate(self, messages, max_new_tokens=350):
            system_message = messages[0]["content"]

            if "rewrite an answer" in system_message.lower():
                return (
                    "Churn increased due to payment failures."
                )

            return """
            {
                "status": "pass",
                "issues": [],
                "final_answer": "Churn increased due to payment failures."
            }
            """

    state = {
        "question": (
            "Why did churn increase among German Enterprise customers in Q2?"
        ),
        "sql_evidence": SQLQueryResult(
            sql="SELECT 52.13 AS churn_percentage",
            rows=[
                {
                    "segment": "Enterprise",
                    "country": "Germany",
                    "churn_percentage": 52.13,
                }
            ],
        ),
        "rag_evidence": RAGResult(
            answer="Payment failures affected customers.",
            sources=[
                RAGSource(
                    document_name="billing_incident.txt",
                    chunk_index=0,
                    text=(
                        "German Enterprise customers experienced elevated "
                        "payment failures during Q2."
                    ),
                    similarity=0.9,
                    reranking_score=0.9,
                )
            ],
        ),
        "draft_answer": (
            "Churn increased due to payment failures."
        ),
    }

    result = verification_node(
        state=state,
        llm=FakeLLM(),
    )

    assert '"status":"revise"' in result["verification"]
    assert "52.13" in result["final_answer"]
    assert not contains_causal_claim(result["final_answer"])
    assert "does not establish a causal relationship" in result["final_answer"]

def test_verification_node_handles_malformed_llm_response_safely():
    from enterprise_ai.orchestration.nodes import verification_node

    class FakeLLM:
        def generate(self, messages, max_new_tokens=350):
            return "This is not valid JSON."

    state = {
        "question": "What happened during the Q2 billing incident?",
        "draft_answer": (
            "The available evidence describes payment failures "
            "during the Q2 billing incident."
        ),
    }

    result = verification_node(
        state=state,
        llm=FakeLLM(),
    )

    assert '"status":"revise"' in result["verification"]
    assert "verification could not be completed reliably" in (
        result["final_answer"].lower()
    )

def test_verification_node_handles_invalid_llm_schema_safely():
    from enterprise_ai.orchestration.nodes import verification_node

    class FakeLLM:
        def generate(self, messages, max_new_tokens=350):
            return """
            {
                "status": "maybe",
                "answer": "This does not match the required schema."
            }
            """

    state = {
        "question": "What happened during the Q2 billing incident?",
        "draft_answer": (
            "The available evidence describes payment failures "
            "during the Q2 billing incident."
        ),
    }

    result = verification_node(
        state=state,
        llm=FakeLLM(),
    )

    assert '"status":"revise"' in result["verification"]
    assert "verification could not be completed reliably" in (
        result["final_answer"].lower()
    )

def test_verification_node_blocks_unsafe_revised_causal_answer():
    from enterprise_ai.orchestration.nodes import verification_node
    from enterprise_ai.rag.models import RAGResult, RAGSource
    from enterprise_ai.sql_agent.models import SQLQueryResult

    class FakeLLM:
        def generate(self, messages, max_new_tokens=350):
            return """
            {
                "status": "revise",
                "issues": ["The original answer needed revision."],
                "final_answer": "Churn increased due to payment failures."
            }
            """

    state = {
        "question": (
            "Why did churn increase among German Enterprise customers in Q2?"
        ),
        "sql_evidence": SQLQueryResult(
            sql="SELECT 52.13 AS churn_percentage",
            rows=[
                {
                    "segment": "Enterprise",
                    "country": "Germany",
                    "churn_percentage": 52.13,
                }
            ],
        ),
        "rag_evidence": RAGResult(
            answer="Payment failures affected customers.",
            sources=[
                RAGSource(
                    document_name="billing_incident.txt",
                    chunk_index=0,
                    text=(
                        "German Enterprise customers experienced elevated "
                        "payment failures during Q2."
                    ),
                    similarity=0.9,
                    reranking_score=0.9,
                )
            ],
        ),
        "draft_answer": "The billing incident caused churn.",
    }

    result = verification_node(
        state=state,
        llm=FakeLLM(),
    )

    assert "52.13" in result["final_answer"]
    assert not contains_causal_claim(result["final_answer"])
    assert "does not establish a causal relationship" in (
        result["final_answer"]
    )

def test_verification_node_preserves_rag_abstention():
    """A grounded RAG abstention must not become a confident final claim."""

    from enterprise_ai.orchestration.nodes import verification_node
    class FakeLLM:
        def generate(self, messages, max_new_tokens):
            return json.dumps(
                {
                    "status": "pass",
                    "issues": [],
                    "final_answer": (
                        "The company did not complete any acquisitions "
                        "in Q2 2025."
                    ),
                }
            )

    state = {
        "question": (
            "What acquisition did the company complete in Q2 2025?"
        ),
        "draft_answer": (
            "The company did not complete any acquisitions in Q2 2025."
        ),
        "rag_evidence": RAGResult(
            answer=(
                "The provided evidence is insufficient to answer "
                "the question."
            ),
            sources=[],
        ),
    }

    result = verification_node(
        state=state,
        llm=FakeLLM(),
    )

    assert "insufficient" in result["final_answer"].lower()

def test_verification_node_preserves_sql_facts_when_rag_abstains():
    """Hybrid answers should keep SQL facts while preserving RAG uncertainty."""

    from enterprise_ai.orchestration.nodes import verification_node

    class FakeLLM:
        def generate(self, messages, max_new_tokens):
            return json.dumps(
                {
                    "status": "pass",
                    "issues": [],
                    "final_answer": (
                        "The provided evidence is insufficient to answer "
                        "the question."
                    ),
                }
            )

    state = {
        "question": (
            "Why did churn increase among German Enterprise customers in Q2?"
        ),
        "sql_evidence": SQLQueryResult(
            sql="SELECT 52.13 AS churn_percentage",
            rows=[
                {
                    "segment": "Enterprise",
                    "country": "Germany",
                    "churn_percentage": 52.13,
                }
            ],
        ),
        "rag_evidence": RAGResult(
            answer=(
                "The provided evidence is insufficient to answer "
                "the question."
            ),
            sources=[],
        ),
        "draft_answer": (
            "The provided evidence is insufficient to answer the question."
        ),
    }

    result = verification_node(
        state=state,
        llm=FakeLLM(),
    )

    assert "52.13" in result["final_answer"]
    assert "insufficient" in result["final_answer"].lower()