from enterprise_ai.evaluation.rag_metrics import (
    abstention_score,
    answer_contains_expected_terms,
)


def test_answer_contains_all_expected_terms():
    answer = (
        "Germany Enterprise customers experienced "
        "a higher concentration of payment failures."
    )

    score = answer_contains_expected_terms(
        answer,
        ["Germany", "Enterprise"],
    )

    assert score == 1.0


def test_answer_missing_expected_term():
    answer = "Germany customers experienced more payment failures."

    score = answer_contains_expected_terms(
        answer,
        ["Germany", "Enterprise"],
    )

    assert score == 0.0


def test_correct_abstention():
    answer = (
        "The provided evidence is insufficient "
        "to answer the question."
    )

    assert abstention_score(
        answer,
        should_abstain=True,
    ) == 1.0


def test_failure_to_abstain():
    answer = "The company's revenue was 2.4 billion euros."

    assert abstention_score(
        answer,
        should_abstain=True,
    ) == 0.0


def test_supported_answer_does_not_abstain():
    answer = (
        "Configuration inconsistencies caused "
        "the payment failures."
    )

    assert abstention_score(
        answer,
        should_abstain=False,
    ) == 1.0


def test_supported_answer_incorrectly_abstains():
    answer = "The evidence is insufficient."

    assert abstention_score(
        answer,
        should_abstain=False,
    ) == 0.0