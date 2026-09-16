import pandas as pd
import pytest

from enterprise_ai.data_generation.generate_customers import (
    generate_customers,
)

from enterprise_ai.validation.customer_validator import (
    validate_customers,
)


def test_customer_generator_returns_expected_rows():

    df = generate_customers(
        num_customers=100
    )

    assert len(df) == 100


def test_customer_ids_are_unique():

    df = generate_customers(
        num_customers=100
    )

    assert df["customer_id"].is_unique


def test_customer_generation_is_reproducible():

    df1 = generate_customers(
        num_customers=100,
        seed=42,
    )

    df2 = generate_customers(
        num_customers=100,
        seed=42,
    )

    pd.testing.assert_frame_equal(
        df1,
        df2,
    )


def test_valid_customers_pass_validation():

    df = generate_customers(
        num_customers=100
    )

    validate_customers(df)


def test_duplicate_customer_ids_fail():

    df = generate_customers(
        num_customers=100
    )

    df.loc[1, "customer_id"] = df.loc[
        0,
        "customer_id",
    ]

    with pytest.raises(
        ValueError,
        match="duplicate",
    ):
        validate_customers(df)


def test_negative_monthly_fee_fails():

    df = generate_customers(
        num_customers=100
    )

    df.loc[0, "monthly_fee"] = -100

    with pytest.raises(ValueError):
        validate_customers(df)


def test_invalid_country_fails():

    df = generate_customers(
        num_customers=100
    )

    df.loc[0, "country"] = "InvalidCountry"

    with pytest.raises(ValueError):
        validate_customers(df)