import numpy as np
import pandas as pd

from enterprise_ai.data_generation.generate_churn import (
    aggregate_ticket_features,
    aggregate_transaction_features,
    build_customer_features,
    generate_churn_labels,
)


def load_test_data():
    customers = pd.read_csv(
        "data/raw/customers.csv"
    )

    transactions = pd.read_csv(
        "data/raw/transactions.csv"
    )

    tickets = pd.read_json(
        "data/raw/support_tickets.json"
    )

    return customers, transactions, tickets


def test_transaction_features_have_valid_rates():
    _, transactions, _ = load_test_data()

    features = aggregate_transaction_features(
        transactions
    )

    assert features["failure_rate"].between(
        0, 1
    ).all()

    assert features["refund_rate"].between(
        0, 1
    ).all()

    assert features["q2_failure_rate"].between(
        0, 1
    ).all()


def test_ticket_features_have_valid_rates():
    _, _, tickets = load_test_data()

    features = aggregate_ticket_features(
        tickets
    )

    assert features["billing_ticket_rate"].between(
        0, 1
    ).all()

    assert features[
        "cancellation_ticket_rate"
    ].between(0, 1).all()

    assert features[
        "q2_billing_ticket_rate"
    ].between(0, 1).all()


def test_customer_feature_table_preserves_customers():
    customers, transactions, tickets = (
        load_test_data()
    )

    transaction_features = (
        aggregate_transaction_features(
            transactions
        )
    )

    ticket_features = (
        aggregate_ticket_features(
            tickets
        )
    )

    features = build_customer_features(
        customers,
        transaction_features,
        ticket_features,
    )

    assert len(features) == len(customers)

    assert features["customer_id"].is_unique


def test_churn_probability_is_valid():
    customers, transactions, tickets = (
        load_test_data()
    )

    transaction_features = (
        aggregate_transaction_features(
            transactions
        )
    )

    ticket_features = (
        aggregate_ticket_features(
            tickets
        )
    )

    features = build_customer_features(
        customers,
        transaction_features,
        ticket_features,
    )

    result = generate_churn_labels(
        features
    )

    assert result[
        "churn_probability"
    ].between(0.01, 0.90).all()

    assert set(
        result["status"].unique()
    ).issubset(
        {"Active", "Churned"}
    )