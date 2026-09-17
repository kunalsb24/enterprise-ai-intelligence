from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def calculate_q2_failure_rates(
    transactions: DataFrame,
    customers: DataFrame,
) -> DataFrame:
    """Compare Q2 payment failure rates for Germany Enterprise vs others."""

    joined = transactions.join(
        customers.select("customer_id", "country", "segment"),
        on="customer_id",
        how="inner",
    )

    q2_transactions = joined.filter(
        (F.col("transaction_date") >= F.lit("2025-04-01"))
        & (F.col("transaction_date") <= F.lit("2025-06-30"))
    )

    classified = q2_transactions.withColumn(
        "customer_group",
        F.when(
            (F.col("country") == "Germany")
            & (F.col("segment") == "Enterprise"),
            "Germany Enterprise",
        ).otherwise("Other Customers"),
    )

    return (
        classified
        .groupBy("customer_group")
        .agg(
            F.count("*").alias("transaction_count"),
            F.sum(
                F.when(
                    F.col("payment_status") == "Failed",
                    1,
                ).otherwise(0)
            ).alias("failed_transactions"),
        )
        .withColumn(
            "failure_rate_pct",
            F.round(
                F.col("failed_transactions")
                / F.col("transaction_count")
                * 100,
                2,
            ),
        )
        .orderBy("customer_group")
    )