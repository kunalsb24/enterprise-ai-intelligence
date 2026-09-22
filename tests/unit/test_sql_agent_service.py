from unittest.mock import MagicMock, patch

from enterprise_ai.sql_agent.models import SQLQueryResult
from enterprise_ai.sql_agent.service import answer_sql_question


def test_answer_sql_question_generates_and_executes_sql():
    llm = MagicMock()
    engine = MagicMock()

    generated_sql = (
        "SELECT COUNT(*) AS customer_count "
        "FROM customers;"
    )

    expected_result = SQLQueryResult(
        sql=generated_sql,
        rows=[
            {
                "customer_count": 10000,
            }
        ],
    )

    with patch(
        "enterprise_ai.sql_agent.service.generate_sql",
        return_value=generated_sql,
    ) as mock_generate:
        with patch(
            "enterprise_ai.sql_agent.service.execute_sql",
            return_value=expected_result,
        ) as mock_execute:
            result = answer_sql_question(
                question="How many customers are there?",
                llm=llm,
                engine=engine,
            )

    mock_generate.assert_called_once_with(
        question="How many customers are there?",
        llm=llm,
    )

    mock_execute.assert_called_once_with(
        sql=generated_sql,
        engine=engine,
    )

    assert result == expected_result