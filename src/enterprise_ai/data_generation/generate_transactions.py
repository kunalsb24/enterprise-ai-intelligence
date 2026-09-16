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

    payment_status = rng.choice(
        [
            "Successful",
            "Failed",
            "Refunded",
        ],
        size=num_transactions,
        p=[
            0.94,
            0.04,
            0.02,
        ],
    )

    transactions = pd.DataFrame(
        {
            "transaction_id": transaction_ids,
            "customer_id": customer_ids,
            "transaction_date": transaction_dates,
            "product": products,
            "channel": channels,
            "amount": amounts,
            "payment_status": payment_status,
        }
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