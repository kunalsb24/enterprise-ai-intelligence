from sqlalchemy import text

from enterprise_ai.database.connection import get_engine


def test_database_connection():
    """Application should connect to PostgreSQL successfully."""

    engine = get_engine()

    with engine.connect() as connection:
        result = connection.execute(
            text("SELECT 1")
        )

        assert result.scalar() == 1