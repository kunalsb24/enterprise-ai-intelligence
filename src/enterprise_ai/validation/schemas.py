from datetime import date
from typing import Literal

from pydantic import BaseModel, Field


class Customer(BaseModel):
    """Schema representing a valid customer record."""

    customer_id: str = Field(
        pattern=r"^CUST_\d{6}$"
    )

    country: Literal[
        "Spain",
        "Germany",
        "France",
        "Italy",
        "Netherlands",
    ]

    segment: Literal[
        "Consumer",
        "SMB",
        "Enterprise",
    ]

    signup_date: date

    contract_type: Literal[
        "Monthly",
        "Annual",
        "Two-Year",
    ]

    monthly_fee: float = Field(
        gt=0,
        le=1000,
    )

    status: Literal[
        "Active",
        "Churned",
    ]