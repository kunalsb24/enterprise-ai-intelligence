from enterprise_ai.orchestration.nodes import router_node
from unittest.mock import Mock, patch

from enterprise_ai.sql_agent.models import SQLQueryResult
from enterprise_ai.rag.models import RAGResult

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