import pandas as pd
from pydantic import ValidationError

from .schemas import Customer


REQUIRED_COLUMNS = {
    "customer_id",
    "country",
    "segment",
    "signup_date",
    "contract_type",
    "monthly_fee",
    "status",
}


def validate_columns(df: pd.DataFrame) -> None:
    """Validate that all required columns exist."""

    missing_columns = REQUIRED_COLUMNS - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )


def validate_nulls(df: pd.DataFrame) -> None:
    """Ensure required fields do not contain null values."""

    null_counts = df[list(REQUIRED_COLUMNS)].isnull().sum()

    invalid_columns = null_counts[null_counts > 0]

    if not invalid_columns.empty:
        raise ValueError(
            f"Null values detected:\n{invalid_columns}"
        )


def validate_unique_customer_ids(df: pd.DataFrame) -> None:
    """Ensure customer IDs are unique."""

    duplicate_count = df["customer_id"].duplicated().sum()

    if duplicate_count > 0:
        raise ValueError(
            f"Found {duplicate_count} duplicate customer IDs"
        )


def validate_records(df: pd.DataFrame) -> None:
    """Validate individual records against the Pydantic schema."""

    errors = []

    for index, row in df.iterrows():

        try:
            Customer.model_validate(
                row.to_dict()
            )

        except ValidationError as exc:
            errors.append(
                {
                    "row": index,
                    "error": str(exc),
                }
            )

    if errors:
        raise ValueError(
            f"{len(errors)} invalid customer records found.\n"
            f"First errors: {errors[:5]}"
        )


def validate_customers(df: pd.DataFrame) -> None:
    """Run the complete customer validation pipeline."""

    validate_columns(df)
    validate_nulls(df)
    validate_unique_customer_ids(df)
    validate_records(df)