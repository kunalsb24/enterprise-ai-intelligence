import json
from pathlib import Path

from enterprise_ai.llm.model import LocalLLM
from enterprise_ai.orchestration.router import decide_route


def load_router_cases(path: str | Path) -> list[dict]:
    """Load routing evaluation cases from JSON."""

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def evaluate_router_case(
    case: dict,
    llm: LocalLLM,
) -> dict:
    """Evaluate one routing question."""

    predicted_route = decide_route(
        question=case["question"],
        llm=llm,
    )

    expected_route = case["expected_route"]

    return {
        "id": case["id"],
        "question": case["question"],
        "expected_route": expected_route,
        "predicted_route": predicted_route,
        "correct": predicted_route == expected_route,
    }


def evaluate_router(
    cases: list[dict],
    llm: LocalLLM,
) -> list[dict]:
    """Evaluate all routing cases."""

    return [
        evaluate_router_case(case, llm)
        for case in cases
    ]

def summarize_router_results(
    results: list[dict],
) -> dict:
    """Calculate overall and per-route routing accuracy."""

    total = len(results)
    correct = sum(
        result["correct"]
        for result in results
    )

    summary = {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "by_route": {},
    }

    for route in ("sql", "rag", "hybrid"):
        route_results = [
            result
            for result in results
            if result["expected_route"] == route
        ]

        route_total = len(route_results)
        route_correct = sum(
            result["correct"]
            for result in route_results
        )

        summary["by_route"][route] = {
            "total": route_total,
            "correct": route_correct,
            "accuracy": (
                route_correct / route_total
                if route_total
                else 0.0
            ),
        }

    return summary