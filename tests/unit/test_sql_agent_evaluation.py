from pathlib import Path

from decimal import Decimal

from enterprise_ai.sql_agent.evaluation import (
    is_numeric_result_correct,
    load_evaluation_cases,
    extract_single_value,
    evaluate_case,
    evaluate_benchmark,
    extract_single_row,
    is_row_result_correct,
)

from unittest.mock import MagicMock, patch

from enterprise_ai.sql_agent.models import SQLQueryResult


def test_load_evaluation_cases():
    path = Path(
        "data/evaluation/sql_agent_questions.json"
    )

    cases = load_evaluation_cases(path)

    assert len(cases) == 7

    assert cases[0]["id"] == "sql_001"
    assert cases[0]["expected_value"] == 10000

    assert cases[1]["id"] == "sql_002"
    assert cases[1]["expected_value"] == 52.13

def test_numeric_result_correctness():
    assert is_numeric_result_correct(
        actual_value=52.13,
        expected_value=52.13,
        tolerance=0.01,
    )

    assert is_numeric_result_correct(
        actual_value=52.129,
        expected_value=52.13,
        tolerance=0.01,
    )

    assert not is_numeric_result_correct(
        actual_value=47.50,
        expected_value=52.13,
        tolerance=0.01,
    )

def test_extract_single_value():
    rows = [
        {
            "churn_percentage": 52.13,
        }
    ]

    value = extract_single_value(rows)

    assert value == 52.13

def test_evaluate_case():
    case = {
        "id": "sql_002",
        "question": (
            "What percentage of German "
            "Enterprise customers churned?"
        ),
        "expected_value": 52.13,
        "tolerance": 0.01,
    }

    llm = MagicMock()
    engine = MagicMock()

    sql_result = SQLQueryResult(
        sql="SELECT 52.13 AS churn_percentage;",
        rows=[
            {
                "churn_percentage": 52.13,
            }
        ],
    )

    with patch(
        "enterprise_ai.sql_agent.evaluation.answer_sql_question",
        return_value=sql_result,
    ):
        evaluation = evaluate_case(
            case=case,
            llm=llm,
            engine=engine,
        )

    assert evaluation["actual_value"] == 52.13
    assert evaluation["expected_value"] == 52.13
    assert evaluation["correct"] is True

def test_evaluate_benchmark():
    cases = [
        {
            "id": "sql_001",
            "question": "Question one",
            "expected_value": 100,
        },
        {
            "id": "sql_002",
            "question": "Question two",
            "expected_value": 50,
        },
    ]

    llm = MagicMock()
    engine = MagicMock()

    mocked_results = [
        {
            "id": "sql_001",
            "question": "Question one",
            "sql": "SELECT 100;",
            "expected_value": 100,
            "actual_value": 100,
            "correct": True,
        },
        {
            "id": "sql_002",
            "question": "Question two",
            "sql": "SELECT 40;",
            "expected_value": 50,
            "actual_value": 40,
            "correct": False,
        },
    ]

    with patch(
        "enterprise_ai.sql_agent.evaluation.evaluate_case",
        side_effect=mocked_results,
    ):
        evaluation = evaluate_benchmark(
            cases=cases,
            llm=llm,
            engine=engine,
        )

    assert evaluation["total_cases"] == 2
    assert evaluation["correct_cases"] == 1
    assert evaluation["accuracy"] == 0.5
    assert len(evaluation["results"]) == 2

def test_evaluate_case_records_failure():
    case = {
        "id": "sql_failure",
        "question": "A question that produces invalid SQL",
        "expected_value": 100,
    }

    llm = MagicMock()
    engine = MagicMock()

    with patch(
        "enterprise_ai.sql_agent.evaluation.answer_sql_question",
        side_effect=ValueError("Invalid SQL syntax."),
    ):
        evaluation = evaluate_case(
            case=case,
            llm=llm,
            engine=engine,
        )

    assert evaluation["correct"] is False
    assert evaluation["actual_value"] is None
    assert evaluation["sql"] is None
    assert evaluation["error"] == "Invalid SQL syntax."

def test_extract_single_row():
    rows = [
        {
            "country": "Spain",
            "churn_rate": 73.41,
        }
    ]

    row = extract_single_row(rows)

    assert row == {
        "country": "Spain",
        "churn_rate": 73.41,
    }

def test_row_result_correctness():
    expected_row = {
        "country": "Spain",
        "churn_rate": 73.41,
    }

    correct_row = {
        "country": "Spain",
        "churn_rate": 73.409,
    }

    wrong_country = {
        "country": "Germany",
        "churn_rate": 73.409,
    }

    assert is_row_result_correct(
        actual_row=correct_row,
        expected_row=expected_row,
        tolerance=0.01,
    )

    assert not is_row_result_correct(
        actual_row=wrong_country,
        expected_row=expected_row,
        tolerance=0.01,
    )

def test_evaluate_case_with_expected_row():
    case = {
        "id": "sql_007",
        "question": (
            "Which country has the highest "
            "customer churn rate?"
        ),
        "expected_row": {
            "country": "Spain",
            "churn_rate": 73.41,
        },
        "tolerance": 0.01,
    }

    llm = MagicMock()
    engine = MagicMock()

    mock_result = SQLQueryResult(
        sql="SELECT country, 73.41 AS churn_rate;",
        rows=[
            {
                "country": "Spain",
                "churn_rate": 73.409,
            }
        ],
    )

    with patch(
        "enterprise_ai.sql_agent.evaluation.answer_sql_question",
        return_value=mock_result,
    ):
        evaluation = evaluate_case(
            case=case,
            llm=llm,
            engine=engine,
        )

    assert evaluation["correct"] is True
    assert evaluation["expected_value"] == {
        "country": "Spain",
        "churn_rate": 73.41,
    }
    assert evaluation["actual_value"] == {
        "country": "Spain",
        "churn_rate": 73.409,
    }
    assert evaluation["error"] is None

def test_row_result_accepts_decimal_numeric_value():
    actual_row = {
        "country": "Spain",
        "churn_rate": Decimal("73.41"),
    }

    expected_row = {
        "country": "Spain",
        "churn_rate": 73.41,
    }

    assert is_row_result_correct(
        actual_row=actual_row,
        expected_row=expected_row,
        tolerance=0.01,
    )

def test_row_result_can_ignore_column_names():
    actual_row = {
        "country": "Spain",
        "churn_rate_percentage": Decimal("73.41"),
    }

    expected_row = {
        "country": "Spain",
        "churn_rate": 73.41,
    }

    assert is_row_result_correct(
        actual_row=actual_row,
        expected_row=expected_row,
        tolerance=0.01,
        match_column_names=False,
    )

def test_evaluate_case_can_ignore_generated_column_aliases(monkeypatch):
    case = {
        "id": "alias_test",
        "question": "Which country has the highest customer churn rate?",
        "expected_row": {
            "country": "Spain",
            "churn_rate": 73.41,
        },
        "tolerance": 0.01,
        "match_column_names": False,
    }

    def fake_answer_sql_question(question, llm, engine):
        return SQLQueryResult(
            sql="SELECT country, 73.41 AS churn_rate_percentage;",
            rows=[
                {
                    "country": "Spain",
                    "churn_rate_percentage": Decimal("73.41"),
                }
            ],
        )

    monkeypatch.setattr(
        "enterprise_ai.sql_agent.evaluation.answer_sql_question",
        fake_answer_sql_question,
    )

    result = evaluate_case(
        case=case,
        llm=None,
        engine=None,
    )

    assert result["correct"] is True