from enterprise_ai.sql_agent.prompts import (
    build_sql_generation_messages,
)


def test_build_sql_generation_messages():
    question = "How many German Enterprise customers churned?"

    messages = build_sql_generation_messages(question)

    assert len(messages) == 2

    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"

    system_message = messages[0]["content"]
    user_message = messages[1]["content"]

    assert "PostgreSQL" in system_message
    assert "SELECT" in system_message
    assert "customers" in system_message
    assert "transactions" in system_message
    assert "support_tickets" in system_message

    assert "DELETE" in system_message
    assert "DROP" in system_message

    assert question in user_message