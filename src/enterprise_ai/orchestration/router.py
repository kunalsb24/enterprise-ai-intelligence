from typing import Literal

from pydantic import BaseModel

from enterprise_ai.llm.model import LocalLLM
from enterprise_ai.orchestration.prompts import build_router_messages
from enterprise_ai.orchestration.state import OrchestrationState, Route


class RoutingDecision(BaseModel):
    """Validated routing decision produced by the router."""

    route: Literal["sql", "rag", "hybrid"]


def route_question(
    state: OrchestrationState,
    route: Route,
) -> OrchestrationState:
    """Store a manually supplied routing decision."""

    decision = RoutingDecision(route=route)

    return {
        "route": decision.route,
    }


def decide_route(
    question: str,
    llm: LocalLLM,
) -> Route:
    """Use the local LLM to decide which route should answer a question."""

    messages = build_router_messages(question)

    raw_route = llm.generate(
        messages=messages,
        max_new_tokens=10,
    )

    normalized_route = raw_route.strip().lower()

    decision = RoutingDecision(route=normalized_route)

    return decision.route