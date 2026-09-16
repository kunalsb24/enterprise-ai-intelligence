import pandas as pd

from enterprise_ai.validation.customer_validator import (
    validate_customers,
)


df = pd.read_csv(
    "data/raw/customers.csv"
)

validate_customers(df)

print(
    f"Validation successful: {len(df):,} customer records"
)