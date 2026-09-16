from pathlib import Path

import numpy as np
import pandas as pd


RANDOM_SEED = 42

CUSTOMERS_PATH = Path("data/raw/customers.csv")
TRANSACTIONS_PATH = Path("data/raw/transactions.csv")
TICKETS_PATH = Path("data/raw/support_tickets.json")
OUTPUT_PATH = Path("data/processed/customers_with_churn.csv")

def aggregate_transaction_features(
    transactions: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate transaction behavior for each customer."""

    transactions = transactions.copy()

    transactions["transaction_date"] = pd.to_datetime(
        transactions["transaction_date"]
    )

    q2_mask = transactions["transaction_date"].between(
        "2025-04-01",
        "2025-06-30",
    )

    transactions["q2_transaction"] = (
        q2_mask.astype(int)
    )

    transactions["q2_failed_transaction"] = (
        q2_mask
        & transactions["payment_status"].eq("Failed")
    ).astype(int)

    features = (
        transactions
        .groupby("customer_id")
        .agg(
            transaction_count=(
                "transaction_id",
                "count",
            ),
            total_spend=(
                "amount",
                "sum",
            ),
            avg_transaction_amount=(
                "amount",
                "mean",
            ),
            failed_transactions=(
                "payment_status",
                lambda x: (x == "Failed").sum(),
            ),
            refunded_transactions=(
                "payment_status",
                lambda x: (x == "Refunded").sum(),
            ),
            q2_transactions=(
                "q2_transaction",
                "sum",
            ),
            q2_failed_transactions=(
                "q2_failed_transaction",
                "sum",
            ),
        )
        .reset_index()
    )

    features["failure_rate"] = (
        features["failed_transactions"]
        / features["transaction_count"]
    )

    features["refund_rate"] = (
        features["refunded_transactions"]
        / features["transaction_count"]
    )

    features["q2_failure_rate"] = (
        features["q2_failed_transactions"]
        / features["q2_transactions"].replace(
            0,
            np.nan,
        )
    ).fillna(0)

    return features

def aggregate_ticket_features(
    tickets: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate support-ticket behavior for each customer."""
    tickets = tickets.copy()

    tickets["created_at"] = pd.to_datetime(
        tickets["created_at"]
    )

    q2_mask = tickets["created_at"].between(
        "2025-04-01",
        "2025-06-30",
    )

    tickets["q2_ticket"] = q2_mask.astype(int)

    tickets["q2_billing_ticket"] = (
        q2_mask
        & tickets["category"].eq("Billing")
    ).astype(int)

    ticket_features = (
        tickets
        .groupby("customer_id")
        .agg(
            total_tickets=(
                "ticket_id",
                "count",
            ),
            billing_tickets=(
                "category",
                lambda x: (x == "Billing").sum(),
            ),
            cancellation_tickets=(
                "category",
                lambda x: (x == "Cancellation").sum(),
            ),
            high_priority_tickets=(
                "priority",
                lambda x: x.isin(
                    ["High", "Critical"]
                ).sum(),
            ),
            q2_tickets=(
                "q2_ticket",
                "sum",
            ),

            q2_billing_tickets=(
                "q2_billing_ticket",
                "sum",
            ),
        )
        .reset_index()
    )

    ticket_features["q2_billing_ticket_rate"] = (
    ticket_features["q2_billing_tickets"]
    / ticket_features["q2_tickets"].replace(0, np.nan)
).fillna(0)

    ticket_features["billing_ticket_rate"] = (
        ticket_features["billing_tickets"]
        / ticket_features["total_tickets"]
    )

    ticket_features["cancellation_ticket_rate"] = (
        ticket_features["cancellation_tickets"]
        / ticket_features["total_tickets"]
    )

    return ticket_features

def build_customer_features(
    customers: pd.DataFrame,
    transaction_features: pd.DataFrame,
    ticket_features: pd.DataFrame,
) -> pd.DataFrame:
    """Combine customer, transaction, and support-ticket features."""

    features = customers.merge(
        transaction_features,
        on="customer_id",
        how="left",
    )

    features = features.merge(
        ticket_features,
        on="customer_id",
        how="left",
    )

    # Customers with no support tickets should have zero ticket activity.
    ticket_columns = [
        "total_tickets",
        "billing_tickets",
        "cancellation_tickets",
        "high_priority_tickets",
        "billing_ticket_rate",
        "cancellation_ticket_rate",
        "q2_tickets",
        "q2_billing_tickets",
        "q2_billing_ticket_rate",
    ]

    features[ticket_columns] = (
        features[ticket_columns]
        .fillna(0)
    )

    return features

def generate_churn_labels(
    features: pd.DataFrame,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Generate churn labels from customer behavioral features."""

    rng = np.random.default_rng(seed)

    result = features.copy()

    # Start every customer with a baseline churn risk.
    churn_probability = np.full(
        len(result),
        0.05,
        dtype=float,
    )

    # Contract behavior:
    # Monthly customers can leave more easily.
    churn_probability += np.where(
        result["contract_type"] == "Monthly",
        0.08,
        0.0,
    )

    # Frequent payment failures increase churn risk.
    churn_probability += (
        result["failure_rate"] * 0.80
    )
    # Recent Q2 payment problems have additional predictive weight.
    churn_probability += (
        result["q2_failure_rate"] * 0.60
    )

    # Refunds are another dissatisfaction signal.
    churn_probability += (
        result["refund_rate"] * 0.30
    )

    # Billing-related support problems increase churn risk.
    churn_probability += (
        result["billing_ticket_rate"] * 0.15
    )

    # Concentrated Q2 billing complaints increase churn risk.
    churn_probability += (
        result["q2_billing_ticket_rate"] * 0.15
    )

    # Cancellation-related tickets are a very strong signal.
    churn_probability += (
        result["cancellation_ticket_rate"] * 0.35
    )

    # Repeated high-priority support issues increase risk.
    churn_probability += np.minimum(
        result["high_priority_tickets"] * 0.015,
        0.10,
    )

    # Keep probability within a sensible range.
    churn_probability = np.clip(
        churn_probability,
        0.01,
        0.90,
    )

    result["churn_probability"] = (
        churn_probability
    )

    random_values = rng.random(
        len(result)
    )

    result["status"] = np.where(
        random_values < result["churn_probability"],
        "Churned",
        "Active",
    )

    return result

def save_churn_data(
    customers_with_churn: pd.DataFrame,
    output_path: Path = OUTPUT_PATH,
) -> None:
    """Save processed customer churn data."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    customers_with_churn.to_csv(
        output_path,
        index=False,
    )

if __name__ == "__main__":

    customers = pd.read_csv(
        CUSTOMERS_PATH
    )

    transactions = pd.read_csv(
        TRANSACTIONS_PATH
    )

    tickets = pd.read_json(
        TICKETS_PATH
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

    customer_features = build_customer_features(
        customers,
        transaction_features,
        ticket_features,
    )

    customers_with_churn = (
    generate_churn_labels(
        customer_features
    )
)
    save_churn_data(
    customers_with_churn
)

    print(
    customers_with_churn[
        [
            "customer_id",
            "country",
            "segment",
            "contract_type",
            "failure_rate",
            "billing_ticket_rate",
            "churn_probability",
            "status",
        ]
    ].head()
)

overall_churn_rate = (
    customers_with_churn["status"]
    .eq("Churned")
    .mean()
)

print(
    f"\nOverall churn rate: "
    f"{overall_churn_rate:.2%}"
)