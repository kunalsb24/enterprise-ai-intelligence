from sqlalchemy import text
from sqlalchemy.engine import Engine

from enterprise_ai.sql_agent.models import SQLQueryResult
from enterprise_ai.sql_agent.validator import validate_sql


def execute_sql(
    sql: str,
    engine: Engine,
) -> SQLQueryResult:
    """Validate and execute SQL using a read-only transaction."""

    validate_sql(sql)

    with engine.connect() as connection:
        connection.execute(
            text("SET TRANSACTION READ ONLY")
        )

        connection.execute(
            text("SET LOCAL statement_timeout = '5s'")
        )

        result = connection.execute(text(sql))

        fetched_rows = list(
            result.mappings().fetchmany(101)
        )

        truncated = len(fetched_rows) > 100

        rows = [
            dict(row)
            for row in fetched_rows[:100]
        ]

        return SQLQueryResult(
            sql=sql,
            rows=rows,
            truncated=truncated,
        )