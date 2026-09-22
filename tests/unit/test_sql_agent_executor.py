from unittest.mock import MagicMock, patch

from enterprise_ai.sql_agent.executor import execute_sql


def test_execute_sql_validates_and_returns_rows():
    engine = MagicMock()
    connection = MagicMock()

    engine.connect.return_value.__enter__.return_value = connection

    query_result = MagicMock()
    query_result.mappings.return_value.fetchmany.return_value = [
        {
            "country": "Germany",
            "customer_count": 376,
        }
    ]

    connection.execute.side_effect = [
        MagicMock(),
        MagicMock(),
        query_result,
    ]

    sql = """
    SELECT country, COUNT(*) AS customer_count
    FROM customers
    GROUP BY country;
    """

    with patch(
        "enterprise_ai.sql_agent.executor.validate_sql"
    ) as mock_validate:
        result = execute_sql(
            sql=sql,
            engine=engine,
        )

    mock_validate.assert_called_once_with(sql)

    assert result.sql == sql
    assert result.rows == [
        {
            "country": "Germany",
            "customer_count": 376,
        }
    ]

    assert result.truncated is False

    query_result.mappings.return_value.fetchmany.assert_called_once_with(
        101
    )

def test_execute_sql_does_not_connect_when_validation_fails():
    engine = MagicMock()

    sql = "DELETE FROM customers;"

    with patch(
        "enterprise_ai.sql_agent.executor.validate_sql",
        side_effect=ValueError("Only SELECT queries are allowed."),
    ):
        try:
            execute_sql(
                sql=sql,
                engine=engine,
            )
        except ValueError:
            pass

    engine.connect.assert_not_called()

def test_execute_sql_marks_large_results_as_truncated():
    engine = MagicMock()
    connection = MagicMock()

    engine.connect.return_value.__enter__.return_value = connection

    query_result = MagicMock()

    database_rows = [
        {
            "customer_id": f"CUST{i:05d}",
        }
        for i in range(101)
    ]

    query_result.mappings.return_value.fetchmany.return_value = (
        database_rows
    )

    connection.execute.side_effect = [
        MagicMock(),
        MagicMock(),
        query_result,
    ]

    sql = "SELECT customer_id FROM customers;"

    with patch(
        "enterprise_ai.sql_agent.executor.validate_sql"
    ):
        result = execute_sql(
            sql=sql,
            engine=engine,
        )

    assert len(result.rows) == 100
    assert result.truncated is True

    query_result.mappings.return_value.fetchmany.assert_called_once_with(
        101
    )