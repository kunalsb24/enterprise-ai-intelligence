import json
from pathlib import Path
from typing import Any
from decimal import Decimal

from sqlalchemy.engine import Engine

from enterprise_ai.llm.model import LocalLLM
from enterprise_ai.sql_agent.service import answer_sql_question




def load_evaluation_cases(
    path: Path,
) -> list[dict[str, Any]]:
    """Load SQL Agent evaluation cases from a JSON file."""

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        cases = json.load(file)

    return cases

def is_numeric_result_correct(
    actual_value: int | float,
    expected_value: int | float,
    tolerance: float = 0.0,
) -> bool:
    """Check whether a numeric SQL result matches the expected value."""

    difference = abs(
        float(actual_value) - float(expected_value)
    )

    return difference <= tolerance

def is_row_result_correct(
    actual_row: dict[str, Any],
    expected_row: dict[str, Any],
    tolerance: float = 0.0,
    match_column_names: bool = True,
) -> bool:
    """Compare a SQL result row with the expected row."""

    if len(actual_row) != len(expected_row):
        return False

    if match_column_names:
        if set(actual_row.keys()) != set(expected_row.keys()):
            return False

        value_pairs = [
            (actual_row[key], expected_value)
            for key, expected_value in expected_row.items()
        ]
    else:
        value_pairs = zip(
            actual_row.values(),
            expected_row.values(),
        )

    for actual_value, expected_value in value_pairs:
        if isinstance(expected_value, (int, float, Decimal)):
            if not isinstance(actual_value, (int, float, Decimal)):
                return False

            if not is_numeric_result_correct(
                actual_value=actual_value,
                expected_value=expected_value,
                tolerance=tolerance,
            ):
                return False

        elif actual_value != expected_value:
            return False

    return True

def extract_single_row(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Extract exactly one row from a SQL result."""

    if len(rows) != 1:
        raise ValueError(
            "Expected exactly one result row."
        )

    return rows[0]

def extract_single_value(
    rows: list[dict[str, Any]],
) -> Any:
    """Extract one value from a single-row, single-column SQL result."""

    if len(rows) != 1:
        raise ValueError(
            "Expected exactly one result row."
        )

    row = rows[0]

    if len(row) != 1:
        raise ValueError(
            "Expected exactly one result column."
        )

    return next(iter(row.values()))

def evaluate_case(
    case: dict[str, Any],
    llm: LocalLLM,
    engine: Engine,
) -> dict[str, Any]:
    """Evaluate one SQL Agent benchmark case."""

    result = None

    try:
        result = answer_sql_question(
            question=case["question"],
            llm=llm,
            engine=engine,
        )

        tolerance = case.get("tolerance", 0.0)

        if "expected_row" in case:
            actual = extract_single_row(
                result.rows
            )
            expected = case["expected_row"]

            correct = is_row_result_correct(
                actual_row=actual,
                expected_row=expected,
                tolerance=tolerance,
                match_column_names=case.get("match_column_names", True),
            )

        else:
            actual = extract_single_value(
                result.rows
            )
            expected = case["expected_value"]

            if actual is None:
                return {
                    "id": case["id"],
                    "question": case["question"],
                    "sql": result.sql,
                    "expected_value": expected,
                    "actual_value": None,
                    "correct": False,
                    "error": "SQL query returned NULL.",
                }

            correct = is_numeric_result_correct(
                actual_value=actual,
                expected_value=expected,
                tolerance=tolerance,
            )

        return {
            "id": case["id"],
            "question": case["question"],
            "sql": result.sql,
            "expected_value": expected,
            "actual_value": actual,
            "correct": correct,
            "error": None,
        }

    except Exception as exc:
        expected = case.get(
            "expected_row",
            case.get("expected_value"),
        )

        return {
            "id": case["id"],
            "question": case["question"],
            "sql": result.sql if result else None,
            "expected_value": expected,
            "actual_value": None,
            "correct": False,
            "error": str(exc),
        }
    
def evaluate_benchmark(
    cases: list[dict[str, Any]],
    llm: LocalLLM,
    engine: Engine,
) -> dict[str, Any]:
    """Evaluate all SQL Agent benchmark cases."""

    results = [
        evaluate_case(
            case=case,
            llm=llm,
            engine=engine,
        )
        for case in cases
    ]

    correct_count = sum(
        result["correct"]
        for result in results
    )

    total_count = len(results)

    accuracy = (
        correct_count / total_count
        if total_count > 0
        else 0.0
    )

    return {
        "total_cases": total_count,
        "correct_cases": correct_count,
        "accuracy": accuracy,
        "results": results,
    }

