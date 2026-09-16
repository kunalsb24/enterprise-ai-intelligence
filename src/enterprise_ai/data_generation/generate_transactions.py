from pathlib import Path

import numpy as np
import pandas as pd


RANDOM_SEED = 42
NUM_TRANSACTIONS = 200_000

OUTPUT_PATH = Path("data/raw/transactions.csv")


def generate_transactions(
    customers: pd.DataFrame,
    num_transactions: int = NUM_TRANSACTIONS,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Generate synthetic transactions linked to customers."""

    rng = np.random.default_rng(seed)

    # ---------------------------------------------------------
    # 1. Generate transaction attributes
    # ---------------------------------------------------------

    customer_ids = rng.choice(
        customers["customer_id"],
        size=num_transactions,
        replace=True,
    )

    transaction_ids = [
        f"TXN_{i:08d}"
        for i in range(1, num_transactions + 1)
    ]

    transaction_dates = pd.to_datetime(
        rng.choice(
            pd.date_range(
                start="2025-01-01",
                end="2025-12-31",
                freq="D",
            ),
            size=num_transactions,
        )
    )

    products = rng.choice(
        [
            "Mobile",
            "Broadband",
            "Cloud",
            "Security",
            "IoT",
        ],
        size=num_transactions,
        p=[
            0.35,
            0.30,
            0.15,
            0.12,
            0.08,
        ],
    )

    channels = rng.choice(
        [
            "Online",
            "Direct Debit",
            "Bank Transfer",
            "Partner",
        ],
        size=num_transactions,
        p=[
            0.35,
            0.40,
            0.15,
            0.10,
        ],
    )

    amounts = np.round(
        rng.lognormal(
            mean=4.2,
            sigma=0.7,
            size=num_transactions,
        ),
        2,
    )

    # ---------------------------------------------------------
    # 2. Create transaction DataFrame
    # ---------------------------------------------------------

    transactions = pd.DataFrame(
        {
            "transaction_id": transaction_ids,
            "customer_id": customer_ids,
            "transaction_date": transaction_dates,
            "product": products,
            "channel": channels,
            "amount": amounts,
        }
    )

    # ---------------------------------------------------------
    # 3. Temporarily attach customer attributes
    # ---------------------------------------------------------

    customer_attributes = customers[
        [
            "customer_id",
            "country",
            "segment",
        ]
    ]

    transactions = transactions.merge(
        customer_attributes,
        on="customer_id",
        how="left",
    )

    # ---------------------------------------------------------
    # 4. Identify Germany Enterprise Q2 incident
    # ---------------------------------------------------------

    incident_mask = (
        (transactions["country"] == "Germany")
        & (transactions["segment"] == "Enterprise")
        & (
            transactions["transaction_date"].between(
                "2025-04-01",
                "2025-06-30",
            )
        )
    )

    # ---------------------------------------------------------
    # 5. Generate normal payment behavior
    # ---------------------------------------------------------

    random_values = rng.random(
        len(transactions)
    )

    # Default = successful
    transactions["payment_status"] = "Successful"

    # 4% failed
    transactions.loc[
        random_values < 0.04,
        "payment_status",
    ] = "Failed"

    # 2% refunded
    transactions.loc[
        (random_values >= 0.04)
        & (random_values < 0.06),
        "payment_status",
    ] = "Refunded"

    # ---------------------------------------------------------
    # 6. Inject Germany Enterprise Q2 billing incident
    # ---------------------------------------------------------

    incident_random = rng.random(
        len(transactions)
    )

    # Start incident population as successful
    transactions.loc[
        incident_mask,
        "payment_status",
    ] = "Successful"

    # 20% failures
    transactions.loc[
        incident_mask
        & (incident_random < 0.20),
        "payment_status",
    ] = "Failed"

    # 4% refunds
    transactions.loc[
        incident_mask
        & (incident_random >= 0.20)
        & (incident_random < 0.24),
        "payment_status",
    ] = "Refunded"

    # ---------------------------------------------------------
    # 7. Remove temporary customer attributes
    # ---------------------------------------------------------

    transactions = transactions.drop(
        columns=[
            "country",
            "segment",
        ]
    )

    return transactions



def save_transactions(
    transactions: pd.DataFrame,
    output_path: Path = OUTPUT_PATH,
) -> None:
    """Save generated transactions."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    transactions.to_csv(
        output_path,
        index=False,
    )


if __name__ == "__main__":

    customers = pd.read_csv(
        "data/raw/customers.csv"
    )

    transactions = generate_transactions(
        customers
    )

    save_transactions(
        transactions
    )

    print(
        f"Generated {len(transactions):,} transactions"
    )

    print(transactions.head())