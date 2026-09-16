from pathlib import Path

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from enterprise_ai.database.connection import get_engine


CUSTOMERS_PATH = Path(
    "data/processed/customers_with_churn.csv"
)

TRANSACTIONS_PATH = Path(
    "data/raw/transactions.csv"
)

SUPPORT_TICKETS_PATH = Path(
    "data/raw/support_tickets.json"
)

def clear_existing_data(engine: Engine) -> None:
    """Remove existing data while preserving database tables and constraints."""

    with engine.begin() as connection:
        connection.execute(
            text(
                """
                TRUNCATE TABLE
                    support_tickets,
                    transactions,
                    customers;
                """
            )
        )

def load_customers(
    engine: Engine,
    path: Path = CUSTOMERS_PATH,
) -> None:
    """Load processed customer data into PostgreSQL."""

    customers = pd.read_csv(path)

    customers["signup_date"] = pd.to_datetime(
        customers["signup_date"]
    ).dt.date

    integer_columns = [
        "transaction_count",
        "failed_transactions",
        "refunded_transactions",
        "q2_transactions",
        "q2_failed_transactions",
        "total_tickets",
        "billing_tickets",
        "cancellation_tickets",
        "high_priority_tickets",
        "q2_tickets",
        "q2_billing_tickets",
    ]

    customers[integer_columns] = (
        customers[integer_columns]
        .fillna(0)
        .astype(int)
    )

    customers.to_sql(
        name="customers",
        con=engine,
        if_exists="append",
        index=False,
        method="multi",
        chunksize=1000,
    )

def load_transactions(
    engine: Engine,
    path: Path = TRANSACTIONS_PATH,
) -> None:
    """Load transaction data into PostgreSQL."""

    transactions = pd.read_csv(path)

    # Convert CSV date strings to Python date objects
    transactions["transaction_date"] = pd.to_datetime(
        transactions["transaction_date"]
    ).dt.date

    transactions.to_sql(
        name="transactions",
        con=engine,
        if_exists="append",
        index=False,
        method="multi",
        chunksize=2000,
    )

def load_support_tickets(
    engine: Engine,
    path: Path = SUPPORT_TICKETS_PATH,
) -> None:
    """Load support ticket data into PostgreSQL."""

    tickets = pd.read_json(path)

    # Convert JSON date values to Python date objects
    tickets["created_at"] = pd.to_datetime(
        tickets["created_at"]
    ).dt.date

    tickets.to_sql(
        name="support_tickets",
        con=engine,
        if_exists="append",
        index=False,
        method="multi",
        chunksize=2000,
    )

def main() -> None:
    engine = get_engine()

    print("Clearing existing database data...")
    clear_existing_data(engine)

    print("Loading customers...")
    load_customers(engine)

    print("Loading transactions...")
    load_transactions(engine)

    print("Loading support tickets...")
    load_support_tickets(engine)

    print("Database load completed successfully.")


if __name__ == "__main__":
    main()
