from datetime import datetime
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator

from enterprise_ai.ingestion.loaders import load_csv

from sqlalchemy import text

from enterprise_ai.database.connection import get_engine

from enterprise_ai.database.load_data import run_full_refresh

CUSTOMERS_PATH = Path("/opt/airflow/data/raw/customers.csv")


def validate_customer_data():
    """Load customer data and perform basic validation checks."""

    customers = load_csv(CUSTOMERS_PATH)

    row_count = len(customers)

    if row_count != 10_000:
        raise ValueError(
            f"Expected 10,000 customers, but found {row_count}"
        )

    if customers["customer_id"].duplicated().any():
        raise ValueError("Duplicate customer IDs detected.")

    print(f"Customer validation passed: {row_count} rows")

def load_database():
    """Run an atomic full refresh of the enterprise PostgreSQL database."""

    engine = get_engine()

    try:
        run_full_refresh(engine)
    finally:
        engine.dispose()

def verify_database():
    """Verify that expected enterprise data exists in PostgreSQL."""

    engine = get_engine()

    with engine.connect() as connection:
        customer_count = connection.execute(
            text("SELECT COUNT(*) FROM customers")
        ).scalar_one()

        transaction_count = connection.execute(
            text("SELECT COUNT(*) FROM transactions")
        ).scalar_one()

        ticket_count = connection.execute(
            text("SELECT COUNT(*) FROM support_tickets")
        ).scalar_one()

    expected_counts = {
        "customers": 10_000,
        "transactions": 200_000,
        "support_tickets": 50_000,
    }

    actual_counts = {
        "customers": customer_count,
        "transactions": transaction_count,
        "support_tickets": ticket_count,
    }

    if actual_counts != expected_counts:
        raise ValueError(
            f"Database verification failed. "
            f"Expected {expected_counts}, found {actual_counts}"
        )

    print(f"Database verification passed: {actual_counts}")

with DAG(
    dag_id="enterprise_data_pipeline",
    description="Enterprise AI data ingestion and validation pipeline",
    start_date=datetime(2025, 1, 1),
    schedule=None,
    catchup=False,
    tags=["enterprise-ai", "data-pipeline"],
) as dag:

    validate_customers = PythonOperator(
        task_id="validate_customers",
        python_callable=validate_customer_data,
    )

    load_database_task = PythonOperator(
        task_id="load_database",
        python_callable=load_database,
    )
    verify_database_task = PythonOperator(
        task_id="verify_database",
        python_callable=verify_database,
    )

    validate_customers >> load_database_task >> verify_database_task