import pandas as pd


customers = pd.read_csv(
    "data/raw/customers.csv"
)

transactions = pd.read_csv(
    "data/raw/transactions.csv",
    parse_dates=["transaction_date"],
)


# Join customer information with transactions
data = transactions.merge(
    customers[
        [
            "customer_id",
            "country",
            "segment",
        ]
    ],
    on="customer_id",
)


# Germany Enterprise transactions during Q2
incident = data[
    (data["country"] == "Germany")
    & (data["segment"] == "Enterprise")
    & (
        data["transaction_date"].between(
            "2025-04-01",
            "2025-06-30",
        )
    )
]


# Everything outside the incident population
normal = data[
    ~data.index.isin(
        incident.index
    )
]


incident_failure_rate = (
    incident["payment_status"]
    .eq("Failed")
    .mean()
)

normal_failure_rate = (
    normal["payment_status"]
    .eq("Failed")
    .mean()
)


print(
    f"Normal failure rate: "
    f"{normal_failure_rate:.2%}"
)

print(
    f"Incident failure rate: "
    f"{incident_failure_rate:.2%}"
)

print(
    f"Failure rate multiplier: "
    f"{incident_failure_rate / normal_failure_rate:.2f}x"
)