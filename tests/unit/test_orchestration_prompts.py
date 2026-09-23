from enterprise_ai.orchestration.prompts import build_router_messages


def test_build_router_messages_contains_question():
    question = "How many customers are there?"

    messages = build_router_messages(question)

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == question


def test_router_prompt_defines_all_routes():
    messages = build_router_messages("Example question")

    system_prompt = messages[0]["content"]

    assert "sql" in system_prompt
    assert "rag" in system_prompt
    assert "hybrid" in system_prompt