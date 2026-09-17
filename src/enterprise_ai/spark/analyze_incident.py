from pathlib import Path

from pyspark.sql import DataFrame

from enterprise_ai.spark.process_transactions import load_transactions
from enterprise_ai.spark.session import get_spark_session

from enterprise_ai.spark.schemas import CUSTOMER_SCHEMA

from enterprise_ai.spark.transformations import calculate_q2_failure_rates

CUSTOMERS_PATH = Path("data/raw/customers.csv")

def load_customers(spark) -> DataFrame:
    """Load raw customer data into a Spark DataFrame."""

    return (
    spark.read
    .option("header", True)
    .schema(CUSTOMER_SCHEMA)
    .csv(str(CUSTOMERS_PATH))
)


def main() -> None:
    spark = get_spark_session()

    transactions = load_transactions(spark)
    customers = load_customers(spark)

    failure_rates = calculate_q2_failure_rates(
        transactions,
        customers,
    )

    print("\nSpark Execution Plan:")
    failure_rates.explain(mode="formatted")

    print("\nQ2 2025 Payment Failure Analysis:")
    failure_rates.show(truncate=False)

    spark.stop()


if __name__ == "__main__":
    main()