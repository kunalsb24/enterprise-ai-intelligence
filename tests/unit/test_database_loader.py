from sqlalchemy import text

from enterprise_ai.database.connection import get_engine
from enterprise_ai.database.load_data import clear_existing_data


def test_clear_existing_data():
    """Full refresh should clear all database tables."""

    engine = get_engine()

    clear_existing_data(engine)

    with engine.connect() as connection:
        customer_count = connection.execute(
            text("SELECT COUNT(*) FROM customers")
        ).scalar_one()

        transaction_count = connection.execute(
            text("SELECT COUNT(*) FROM transactions")
        ).scalar_one()

        ticket_count = connection.execute(
            text("SELECT COUNT(*) FROM support_tickets")
        ).scalar_one()

    assert customer_count == 0
    assert transaction_count == 0
    assert ticket_count == 0