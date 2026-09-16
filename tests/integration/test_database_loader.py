from sqlalchemy import text

from enterprise_ai.database.connection import get_engine
from enterprise_ai.database.load_data import (
    clear_existing_data,
    load_customers,
    load_transactions,
    load_support_tickets,
)


def get_table_count(connection, table_name: str) -> int:
    """Return the number of rows in a database table."""

    result = connection.execute(
        text(f"SELECT COUNT(*) FROM {table_name}")
    )

    return result.scalar_one()


def test_full_database_load():
    """Full database load should populate all tables with expected row counts."""

    engine = get_engine()

    # Start from a clean database.
    clear_existing_data(engine)

    # Load in foreign-key dependency order.
    load_customers(engine)
    load_transactions(engine)
    load_support_tickets(engine)

    with engine.connect() as connection:
        assert get_table_count(connection, "customers") == 10_000
        assert get_table_count(connection, "transactions") == 200_000
        assert get_table_count(connection, "support_tickets") == 50_000