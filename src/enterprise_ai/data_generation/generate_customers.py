from pathlib import Path

import numpy as np
import pandas as pd


RANDOM_SEED = 42
NUM_CUSTOMERS = 10_000

OUTPUT_PATH = Path("data/raw/customers.csv")


def generate_customers(
    num_customers: int = NUM_CUSTOMERS,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Generate synthetic telecom customer data."""

    rng = np.random.default_rng(seed)

    customer_ids = [
        f"CUST_{i:06d}"
        for i in range(1, num_customers + 1)
    ]

    countries = rng.choice(
        ["Spain", "Germany", "France", "Italy", "Netherlands"],
        size=num_customers,
        p=[0.30, 0.25, 0.20, 0.15, 0.10],
    )

    segments = rng.choice(
        ["Consumer", "SMB", "Enterprise"],
        size=num_customers,
        p=[0.60, 0.25, 0.15],
    )

    contract_types = rng.choice(
        ["Monthly", "Annual", "Two-Year"],
        size=num_customers,
        p=[0.50, 0.35, 0.15],
    )

    signup_dates = pd.to_datetime(
        rng.choice(
            pd.date_range(
                start="2021-01-01",
                end="2025-12-31",
                freq="D",
            ),
            size=num_customers,
        )
    )

    monthly_fees = np.round(
        rng.uniform(20, 500, size=num_customers),
        2,
    )

    status = rng.choice(
        ["Active", "Churned"],
        size=num_customers,
        p=[0.82, 0.18],
    )

    customers = pd.DataFrame(
        {
            "customer_id": customer_ids,
            "country": countries,
            "segment": segments,
            "signup_date": signup_dates,
            "contract_type": contract_types,
            "monthly_fee": monthly_fees,
            "status": status,
        }
    )

    return customers


def save_customers(
    customers: pd.DataFrame,
    output_path: Path = OUTPUT_PATH,
) -> None:
    """Save generated customer data to CSV."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    customers.to_csv(
        output_path,
        index=False,
    )


if __name__ == "__main__":
    customers_df = generate_customers()

    save_customers(customers_df)

    print(f"Generated {len(customers_df):,} customers")
    print(customers_df.head())