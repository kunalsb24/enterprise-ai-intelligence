from datetime import date

from enterprise_ai.spark.transformations import calculate_q2_failure_rates


def test_calculate_q2_failure_rates(spark):

    customers = spark.createDataFrame(
        [
            ("CUST_001", "Germany", "Enterprise"),
            ("CUST_002", "Spain", "Consumer"),
        ],
        ["customer_id", "country", "segment"],
    )

    transactions = spark.createDataFrame(
        [
            ("TXN_001", "CUST_001", date(2025, 4, 10), "Failed"),
            ("TXN_002", "CUST_001", date(2025, 5, 10), "Successful"),
            ("TXN_003", "CUST_002", date(2025, 4, 15), "Successful"),
            ("TXN_004", "CUST_002", date(2025, 6, 15), "Successful"),

            # Outside Q2 — should be ignored
            ("TXN_005", "CUST_001", date(2025, 7, 10), "Failed"),
        ],
        [
            "transaction_id",
            "customer_id",
            "transaction_date",
            "payment_status",
        ],
    )

    result = calculate_q2_failure_rates(
        transactions,
        customers,
    )

    rows = {
        row["customer_group"]: row
        for row in result.collect()
    }

    germany = rows["Germany Enterprise"]
    others = rows["Other Customers"]

    assert germany["transaction_count"] == 2
    assert germany["failed_transactions"] == 1
    assert germany["failure_rate_pct"] == 50.0

    assert others["transaction_count"] == 2
    assert others["failed_transactions"] == 0
    assert others["failure_rate_pct"] == 0.0

    