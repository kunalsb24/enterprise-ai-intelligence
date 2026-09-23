from enterprise_ai.orchestration.evaluation import (
    evaluate_router,
    evaluate_router_case,
)

from enterprise_ai.orchestration.evaluation import summarize_router_results


class FakeLLM:
    def generate(self, messages, max_new_tokens=200):
        return "sql"


def test_evaluate_router_case_marks_correct_prediction():
    case = {
        "id": "route_test",
        "question": "How many customers are there?",
        "expected_route": "sql",
    }

    result = evaluate_router_case(
        case,
        FakeLLM(),
    )

    assert result["expected_route"] == "sql"
    assert result["predicted_route"] == "sql"
    assert result["correct"] is True


def test_evaluate_router_case_marks_incorrect_prediction():
    case = {
        "id": "route_test",
        "question": "What happened during the incident?",
        "expected_route": "rag",
    }

    result = evaluate_router_case(
        case,
        FakeLLM(),
    )

    assert result["expected_route"] == "rag"
    assert result["predicted_route"] == "sql"
    assert result["correct"] is False


def test_evaluate_router_runs_all_cases():
    cases = [
        {
            "id": "route_001",
            "question": "How many customers are there?",
            "expected_route": "sql",
        },
        {
            "id": "route_002",
            "question": "Which segment has the most customers?",
            "expected_route": "sql",
        },
    ]

    results = evaluate_router(
        cases,
        FakeLLM(),
    )

    assert len(results) == 2
    assert all(result["correct"] for result in results)


def test_summarize_router_results_calculates_accuracy():
    results = [
        {
            "expected_route": "sql",
            "predicted_route": "sql",
            "correct": True,
        },
        {
            "expected_route": "rag",
            "predicted_route": "rag",
            "correct": True,
        },
        {
            "expected_route": "hybrid",
            "predicted_route": "rag",
            "correct": False,
        },
    ]

    summary = summarize_router_results(results)

    assert summary["total"] == 3
    assert summary["correct"] == 2
    assert summary["accuracy"] == 2 / 3

    assert summary["by_route"]["sql"]["accuracy"] == 1.0
    assert summary["by_route"]["rag"]["accuracy"] == 1.0
    assert summary["by_route"]["hybrid"]["accuracy"] == 0.0
