from unittest.mock import MagicMock, patch

from enterprise_ai.sql_agent.generator import generate_sql


@patch(
    "enterprise_ai.sql_agent.generator."
    "build_sql_generation_messages"
)
def test_generate_sql(mock_build_messages):
    messages = [
        {
            "role": "system",
            "content": "Generate PostgreSQL.",
        },
        {
            "role": "user",
            "content": "How many customers churned?",
        },
    ]

    mock_build_messages.return_value = messages

    llm = MagicMock()
    llm.generate.return_value = (
        "  SELECT COUNT(*) FROM customers "
        "WHERE status = 'Churned';  "
    )

    result = generate_sql(
        question="How many customers churned?",
        llm=llm,
    )

    assert result == (
        "SELECT COUNT(*) FROM customers "
        "WHERE status = 'Churned';"
    )

    mock_build_messages.assert_called_once_with(
        "How many customers churned?"
    )

    llm.generate.assert_called_once_with(
        messages=messages,
        max_new_tokens=300,
    )