from enterprise_ai.orchestration.verification import (
    contains_causal_claim,
    primary_sources_support_causation,
    format_sql_facts,
)

def test_contains_causal_claim_detects_strong_causal_language():
    assert contains_causal_claim(
        "Churn increased due to payment failures."
    )

    assert contains_causal_claim(
        "The billing incident led to increased churn."
    )

    assert contains_causal_claim(
        "Payment failures caused customer churn."
    )


def test_contains_causal_claim_allows_non_causal_language():
    assert not contains_causal_claim(
        "Churn and payment failures increased during the same period."
    )

    assert not contains_causal_claim(
        "Payment failures were associated with elevated support activity."
    )

    assert not contains_causal_claim(
        "The available evidence does not establish causation."
    )

def test_primary_sources_support_causation_when_explicit():
    source_texts = [
        "Payment failures increased during Q2.",
        "The billing incident caused customer churn.",
    ]

    assert primary_sources_support_causation(source_texts)


def test_primary_sources_do_not_support_causation_from_cooccurrence():
    source_texts = [
        "Payment failures increased during Q2.",
        "Billing support requests increased during the same period.",
        "Enterprise customers in Germany were particularly affected.",
    ]

    assert not primary_sources_support_causation(source_texts)


def test_format_sql_facts_formats_rows():
    rows = [
        {
            "segment": "Enterprise",
            "country": "Germany",
            "churn_percentage": 52.13,
        }
    ]

    result = format_sql_facts(rows)

    assert result == (
        "segment=Enterprise, country=Germany, churn_percentage=52.13"
    )


def test_format_sql_facts_handles_empty_rows():
    assert format_sql_facts([]) == ""