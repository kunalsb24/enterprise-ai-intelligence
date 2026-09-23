from enterprise_ai.orchestration.router import route_question

import pytest
from pydantic import ValidationError

from enterprise_ai.orchestration.router import RoutingDecision


def test_routing_decision_rejects_invalid_route():
    with pytest.raises(ValidationError):
        RoutingDecision(route="database")


def test_route_question_returns_selected_route():
    state = {
        "question": "How many customers are there?"
    }

    result = route_question(state, "sql")

    assert result == {
        "route": "sql",
    }

from enterprise_ai.orchestration.router import decide_route


class FakeLLM:
    def generate(self, messages, max_new_tokens=200):
        return "SQL"


def test_decide_route_normalizes_and_validates_llm_output():
    llm = FakeLLM()

    result = decide_route(
        "How many customers are there?",
        llm,
    )

    assert result == "sql"
