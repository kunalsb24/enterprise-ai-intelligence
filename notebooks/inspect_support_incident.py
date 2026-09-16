import pandas as pd


customers = pd.read_csv(
    "data/raw/customers.csv"
)

tickets = pd.read_json(
    "data/raw/support_tickets.json"
)

tickets["created_at"] = pd.to_datetime(
    tickets["created_at"]
)

# Attach customer attributes for analysis
data = tickets.merge(
    customers[
        [
            "customer_id",
            "country",
            "segment",
        ]
    ],
    on="customer_id",
    how="left",
)

# Germany Enterprise customers during Q2
incident_mask = (
    (data["country"] == "Germany")
    & (data["segment"] == "Enterprise")
    & data["created_at"].between(
        "2025-04-01",
        "2025-06-30",
    )
)

incident = data[incident_mask]
normal = data[~incident_mask]

normal_billing_rate = (
    normal["category"]
    .eq("Billing")
    .mean()
)

incident_billing_rate = (
    incident["category"]
    .eq("Billing")
    .mean()
)

print(
    f"Normal billing-ticket rate: "
    f"{normal_billing_rate:.2%}"
)

print(
    f"Incident billing-ticket rate: "
    f"{incident_billing_rate:.2%}"
)

print(
    f"Billing-ticket multiplier: "
    f"{incident_billing_rate / normal_billing_rate:.2f}x"
)

print(
    f"Incident tickets: "
    f"{len(incident):,}"
)