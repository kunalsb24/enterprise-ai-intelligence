import pytest
import enterprise_ai.database.load_data as load_data_module
from sqlalchemy import create_engine, text

from enterprise_ai.database.load_data import run_full_refresh
from enterprise_ai.database.load_data import clear_existing_data


TEST_DATABASE_URL = (
    "postgresql+psycopg://enterprise_user:enterprise_password"
    "@localhost:5433/enterprise_ai_test"
)


@pytest.fixture(scope="module")
def test_engine():
    """Provide a database engine connected to the isolated test database."""

    engine = create_engine(TEST_DATABASE_URL)

    yield engine

    engine.dispose()


def get_table_count(connection, table_name: str) -> int:
    """Return the number of rows in a database table."""

    result = connection.execute(
        text(f"SELECT COUNT(*) FROM {table_name}")
    )

    return result.scalar_one()


def test_full_database_load(test_engine):
    """Full refresh should populate the isolated test database."""

    run_full_refresh(test_engine)

    with test_engine.connect() as connection:
        assert get_table_count(connection, "customers") == 10_000
        assert get_table_count(connection, "transactions") == 200_000
        assert get_table_count(connection, "support_tickets") == 50_000


def test_transaction_rolls_back_on_failure(test_engine):
    """A failed transaction should preserve the existing database state."""

    with test_engine.connect() as connection:
        counts_before = {
            "customers": get_table_count(connection, "customers"),
            "transactions": get_table_count(connection, "transactions"),
            "support_tickets": get_table_count(connection, "support_tickets"),
        }

    try:
        with test_engine.begin() as connection:
            clear_existing_data(connection)

            # Simulate an unexpected failure during the refresh.
            raise RuntimeError("Simulated pipeline failure")

    except RuntimeError:
        pass

    with test_engine.connect() as connection:
        counts_after = {
            "customers": get_table_count(connection, "customers"),
            "transactions": get_table_count(connection, "transactions"),
            "support_tickets": get_table_count(connection, "support_tickets"),
        }

    assert counts_after == counts_before


def test_full_refresh_rolls_back_when_loader_fails(
    test_engine,
    monkeypatch,
):
    """Full refresh should roll back if a loader fails."""

    with test_engine.connect() as connection:
        counts_before = {
            "customers": get_table_count(connection, "customers"),
            "transactions": get_table_count(connection, "transactions"),
            "support_tickets": get_table_count(
                connection,
                "support_tickets",
            ),
        }

    def simulate_transaction_failure(*args, **kwargs):
        raise RuntimeError("Simulated transaction loading failure")

    monkeypatch.setattr(
        load_data_module,
        "load_transactions",
        simulate_transaction_failure,
    )

    with pytest.raises(
        RuntimeError,
        match="Simulated transaction loading failure",
    ):
        load_data_module.run_full_refresh(test_engine)

    with test_engine.connect() as connection:
        counts_after = {
            "customers": get_table_count(connection, "customers"),
            "transactions": get_table_count(connection, "transactions"),
            "support_tickets": get_table_count(
                connection,
                "support_tickets",
            ),
        }

    assert counts_after == counts_before