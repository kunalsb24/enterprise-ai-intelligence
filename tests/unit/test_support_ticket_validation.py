import pandas as pd

from enterprise_ai.data_generation.generate_support_tickets import (
    generate_support_tickets,
)


def test_support_ticket_count():
    customers = pd.read_csv(
        "data/raw/customers.csv"
    )

    tickets = generate_support_tickets(
        customers,
        num_tickets=1000,
    )

    assert len(tickets) == 1000


def test_ticket_ids_are_unique():
    customers = pd.read_csv(
        "data/raw/customers.csv"
    )

    tickets = generate_support_tickets(
        customers,
        num_tickets=1000,
    )

    assert tickets["ticket_id"].is_unique


def test_all_tickets_reference_valid_customers():
    customers = pd.read_csv(
        "data/raw/customers.csv"
    )

    tickets = generate_support_tickets(
        customers,
        num_tickets=1000,
    )

    assert tickets["customer_id"].isin(
        customers["customer_id"]
    ).all()


def test_ticket_categories_are_valid():
    customers = pd.read_csv(
        "data/raw/customers.csv"
    )

    tickets = generate_support_tickets(
        customers,
        num_tickets=1000,
    )

    valid_categories = {
        "Technical",
        "Billing",
        "Account",
        "Service",
        "Cancellation",
    }

    assert tickets["category"].isin(
        valid_categories
    ).all()


def test_ticket_descriptions_are_not_empty():
    customers = pd.read_csv(
        "data/raw/customers.csv"
    )

    tickets = generate_support_tickets(
        customers,
        num_tickets=1000,
    )

    assert tickets["description"].notna().all()

    assert (
        tickets["description"]
        .str.strip()
        .ne("")
        .all()
    )