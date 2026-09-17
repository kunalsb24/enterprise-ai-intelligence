from pathlib import Path

from pyspark.sql import DataFrame, SparkSession

from enterprise_ai.spark.session import get_spark_session

from enterprise_ai.spark.schemas import TRANSACTION_SCHEMA

TRANSACTIONS_PATH = Path("data/raw/transactions.csv")


def load_transactions(
    spark: SparkSession,
    path: Path = TRANSACTIONS_PATH,
) -> DataFrame:
    """Load raw transaction data into a Spark DataFrame."""

    return (
    spark.read
    .option("header", True)
    .schema(TRANSACTION_SCHEMA)
    .csv(str(path))
)

def main() -> None:
    spark = get_spark_session()

    transactions = load_transactions(spark)

    print("\nTransaction schema:")
    transactions.printSchema()

    print("\nSample transactions:")
    transactions.show(5, truncate=False)

    print("\nTransaction count:")
    print(transactions.count())

    print("\nNumber of partitions:")
    print(transactions.rdd.getNumPartitions())

    spark.stop()


if __name__ == "__main__":
    main()