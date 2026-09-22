from enterprise_ai.llm.model import LocalLLM
from enterprise_ai.sql_agent.prompts import (
    build_sql_generation_messages,
)


def generate_sql(
    question: str,
    llm: LocalLLM,
) -> str:
    """Generate a SQL query for a business question."""

    messages = build_sql_generation_messages(question)

    sql = llm.generate(
        messages=messages,
        max_new_tokens=300,
    )

    return sql.strip()