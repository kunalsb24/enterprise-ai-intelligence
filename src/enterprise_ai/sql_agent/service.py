from sqlalchemy.engine import Engine

from enterprise_ai.llm.model import LocalLLM
from enterprise_ai.sql_agent.executor import execute_sql
from enterprise_ai.sql_agent.generator import generate_sql
from enterprise_ai.sql_agent.models import SQLQueryResult


def answer_sql_question(
    question: str,
    llm: LocalLLM,
    engine: Engine,
) -> SQLQueryResult:
    """Generate and safely execute SQL for a business question."""

    sql = generate_sql(
        question=question,
        llm=llm,
    )

    return execute_sql(
        sql=sql,
        engine=engine,
    )