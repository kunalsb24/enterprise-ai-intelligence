from pyspark.sql.types import (
    DateType,
    DoubleType,
    StringType,
    StructField,
    StructType,
)


TRANSACTION_SCHEMA = StructType(
    [
        StructField("transaction_id", StringType(), nullable=False),
        StructField("customer_id", StringType(), nullable=False),
        StructField("transaction_date", DateType(), nullable=False),
        StructField("product", StringType(), nullable=False),
        StructField("channel", StringType(), nullable=False),
        StructField("amount", DoubleType(), nullable=False),
        StructField("payment_status", StringType(), nullable=False),
    ]
)


CUSTOMER_SCHEMA = StructType(
    [
        StructField("customer_id", StringType(), nullable=False),
        StructField("country", StringType(), nullable=False),
        StructField("segment", StringType(), nullable=False),
        StructField("contract_type", StringType(), nullable=False),
        StructField("signup_date", DateType(), nullable=False),
        StructField("monthly_fee", DoubleType(), nullable=False),
    ]
)