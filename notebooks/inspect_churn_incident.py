import pandas as pd


customers = pd.read_csv(
    "data/processed/customers_with_churn.csv"
)

# Customers affected by our synthetic incident
incident_mask = (
    (customers["country"] == "Germany")
    & (customers["segment"] == "Enterprise")
)

incident_customers = customers[
    incident_mask
]

other_customers = customers[
    ~incident_mask
]

incident_churn_rate = (
    incident_customers["status"]
    .eq("Churned")
    .mean()
)

normal_churn_rate = (
    other_customers["status"]
    .eq("Churned")
    .mean()
)

print(
    f"Other customers churn rate: "
    f"{normal_churn_rate:.2%}"
)

print(
    f"Germany Enterprise churn rate: "
    f"{incident_churn_rate:.2%}"
)

print(
    f"Churn multiplier: "
    f"{incident_churn_rate / normal_churn_rate:.2f}x"
)

print(
    f"Germany Enterprise customers: "
    f"{len(incident_customers):,}"
)