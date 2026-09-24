from unittest.mock import Mock, patch

from enterprise_ai.orchestration.graph import build_router_graph
from enterprise_ai.rag.models import RAGResult
from enterprise_ai.sql_agent.models import SQLQueryResult


class FakeLLM:
    def __init__(self, route: str):
        self.route = route

    def generate(self, messages, max_new_tokens=200):
        system_message = messages[0]["content"].lower()

        if "verification component" in system_message:
            return """
            {
                "status": "pass",
                "issues": [],
                "final_answer": "Verified grounded answer."
            }
            """

        if "evidence synthesis component" in system_message:
            return "Synthesized grounded answer."

        return self.route


def test_graph_routes_to_sql_branch():
    expected_result = SQLQueryResult(
        sql="SELECT COUNT(*) AS customer_count FROM customers",
        rows=[
            {"customer_count": 10000}
        ],
        truncated=False,
    )

    with patch(
        "enterprise_ai.orchestration.nodes.answer_sql_question",
        return_value=expected_result,
    ):
        graph = build_router_graph(
            FakeLLM("sql"),
            Mock(),  # database engine
            Mock(),  # embedding model
            Mock(),  # reranking model
        )

        result = graph.invoke(
            {"question": "How many customers are there?"}
        )

    assert result["route"] == "sql"
    assert result["sql_evidence"] == expected_result
    assert "rag_evidence" not in result


def test_graph_routes_to_rag_branch():
    expected_result = RAGResult(
        answer="The incident involved payment failures.",
        sources=[],
    )

    with patch(
        "enterprise_ai.orchestration.nodes.answer_question",
        return_value=expected_result,
    ):
        graph = build_router_graph(
            FakeLLM("rag"),
            Mock(),  # database engine
            Mock(),  # embedding model
            Mock(),  # reranking model
        )

        result = graph.invoke(
            {"question": "What happened during the incident?"}
        )

    assert result["route"] == "rag"
    assert result["rag_evidence"] == expected_result
    assert "sql_evidence" not in result


def test_graph_routes_to_hybrid_branch():
    expected_sql_result = SQLQueryResult(
        sql="SELECT 1",
        rows=[
            {"value": 1}
        ],
        truncated=False,
    )

    expected_rag_result = RAGResult(
        answer="The incident involved payment failures.",
        sources=[],
    )

    with (
        patch(
            "enterprise_ai.orchestration.nodes.answer_sql_question",
            return_value=expected_sql_result,
        ) as mock_sql,
        patch(
            "enterprise_ai.orchestration.nodes.answer_question",
            return_value=expected_rag_result,
        ) as mock_rag,
    ):
        graph = build_router_graph(
            FakeLLM("hybrid"),
            Mock(),  # database engine
            Mock(),  # embedding model
            Mock(),  # reranking model
        )

        result = graph.invoke(
            {
                "question":
                "Why did churn increase during the incident?"
            }
        )

    assert result["route"] == "hybrid"
    assert result["sql_evidence"] == expected_sql_result
    assert result["rag_evidence"] == expected_rag_result

    mock_sql.assert_called_once()
    mock_rag.assert_called_once()