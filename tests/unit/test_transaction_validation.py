import pandas as pd


def test_all_transactions_reference_valid_customers():

    customers = pd.read_csv(
        "data/raw/customers.csv"
    )

    transactions = pd.read_csv(
        "data/raw/transactions.csv"
    )

    invalid = ~transactions["customer_id"].isin(
        customers["customer_id"]
    )

    assert invalid.sum() == 0