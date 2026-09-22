from enterprise_ai.sql_agent.connection import get_sql_agent_engine
from enterprise_ai.sql_agent.executor import execute_sql

import pytest
from sqlalchemy import text
from sqlalchemy.exc import ProgrammingError

def test_execute_sql_against_real_database():
    engine = get_sql_agent_engine()

    result = execute_sql(
        sql="SELECT COUNT(*) AS customer_count FROM customers;",
        engine=engine,
    )

    assert len(result.rows) == 1
    assert result.rows[0]["customer_count"] == 10000

def test_reader_cannot_access_document_chunks():
    engine = get_sql_agent_engine()

    with engine.connect() as connection:
        with pytest.raises(ProgrammingError):
            connection.execute(
                text(
                    "SELECT COUNT(*) "
                    "FROM document_chunks;"
                )
            )

def test_reader_cannot_update_customers():
    engine = get_sql_agent_engine()

    with engine.connect() as connection:
        with pytest.raises(ProgrammingError):
            connection.execute(
                text(
                    """
                    UPDATE customers
                    SET status = 'Churned'
                    WHERE customer_id = 'CUST00001';
                    """
                )
            )