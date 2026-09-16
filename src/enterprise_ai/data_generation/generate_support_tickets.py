from pathlib import Path
import json

import numpy as np
import pandas as pd


RANDOM_SEED = 42
NUM_TICKETS = 50_000

OUTPUT_PATH = Path("data/raw/support_tickets.json")

def generate_description(
    category: str,
    rng: np.random.Generator,
) -> str:
    """Generate a support ticket description."""

    descriptions = {
        "Billing": [
            "Customer reports an unexpected charge on the latest invoice.",
            "Payment was declined despite valid account details.",
            "Customer reports being charged twice for the same service.",
            "Invoice amount does not match the agreed contract.",
            "Automatic payment failed and customer requests assistance.",
        ],
        "Technical": [
            "Customer reports intermittent service connectivity.",
            "Application is unavailable for multiple users.",
            "Customer reports slow network performance.",
            "Unable to access the service dashboard.",
            "Service stopped working after a recent update.",
        ],
        "Account": [
            "Customer cannot access their account.",
            "User requests an account information update.",
            "Customer reports problems resetting their password.",
            "Account permissions appear to be incorrect.",
        ],
        "Service": [
            "Customer reports degraded service quality.",
            "Requested service has not been activated.",
            "Customer asks about availability of additional services.",
            "Service configuration does not match customer requirements.",
        ],
        "Cancellation": [
            "Customer requests cancellation of the current contract.",
            "Customer is considering leaving due to service issues.",
            "Customer requests information about contract termination.",
        ],
    }

    return str(
        rng.choice(
            descriptions[category]
        )
    )

def generate_support_tickets(
    customers: pd.DataFrame,
    num_tickets: int = NUM_TICKETS,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Generate synthetic customer support tickets."""

    rng = np.random.default_rng(seed)

    # Select customers who created tickets
    customer_ids = rng.choice(
        customers["customer_id"],
        size=num_tickets,
        replace=True,
    )

    ticket_ids = [
        f"TKT_{i:07d}"
        for i in range(1, num_tickets + 1)
    ]

    created_dates = pd.to_datetime(
        rng.choice(
            pd.date_range(
                start="2025-01-01",
                end="2025-12-31",
                freq="D",
            ),
            size=num_tickets,
        )
    )

    priorities = rng.choice(
        ["Low", "Medium", "High", "Critical"],
        size=num_tickets,
        p=[0.25, 0.45, 0.25, 0.05],
    )

    tickets = pd.DataFrame(
        {
            "ticket_id": ticket_ids,
            "customer_id": customer_ids,
            "created_at": created_dates,
            "priority": priorities,
        }
    )

    # Temporarily attach customer information
    customer_attributes = customers[
        [
            "customer_id",
            "country",
            "segment",
        ]
    ]

    tickets = tickets.merge(
        customer_attributes,
        on="customer_id",
        how="left",
    )

    categories = rng.choice(
    [
        "Technical",
        "Billing",
        "Account",
        "Service",
        "Cancellation",
    ],
    size=len(tickets),
    p=[
        0.35,
        0.15,
        0.20,
        0.20,
        0.10,
    ],
)

    tickets["category"] = categories

    incident_mask = (
    (tickets["country"] == "Germany")
    & (tickets["segment"] == "Enterprise")
    & (
        tickets["created_at"].between(
            "2025-04-01",
            "2025-06-30",
        )
    )
)
    incident_random = rng.random(
    len(tickets)
)
    tickets.loc[
    incident_mask & (incident_random < 0.65),
    "category",
] = "Billing"

    tickets["description"] = [
    generate_description(
        category,
        rng,
    )
    for category in tickets["category"]
]

    # Remove temporary customer attributes
    tickets = tickets.drop(
        columns=["country", "segment"]
    )

    return tickets

def save_support_tickets(
    tickets: pd.DataFrame,
    output_path: Path = OUTPUT_PATH,
) -> None:
    """Save support tickets as JSON."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    records = tickets.copy()

    records["created_at"] = (
        records["created_at"]
        .dt.strftime("%Y-%m-%d")
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            records.to_dict(orient="records"),
            file,
            indent=2,
        )

if __name__ == "__main__":

        customers = pd.read_csv(
            "data/raw/customers.csv"
        )

        tickets = generate_support_tickets(
            customers
        )

        save_support_tickets(
            tickets
        )

        print(
            f"Generated {len(tickets):,} support tickets"
        )

        print(tickets.head())