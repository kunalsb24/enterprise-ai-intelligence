import os

from dotenv import load_dotenv
from sqlalchemy import URL, create_engine
from sqlalchemy.engine import Engine


def get_sql_agent_engine() -> Engine:
    """Create a PostgreSQL engine for the restricted SQL Agent user."""

    load_dotenv()

    host = os.getenv("SQL_AGENT_DATABASE_HOST", "localhost")
    port = os.getenv("SQL_AGENT_DATABASE_PORT", "5433")
    database = os.getenv("SQL_AGENT_DATABASE_NAME", "enterprise_ai")
    user = os.getenv("SQL_AGENT_DATABASE_USER")
    password = os.getenv("SQL_AGENT_DATABASE_PASSWORD")

    if not user or not password:
        raise ValueError(
            "SQL Agent database credentials are not configured."
        )

    database_url = URL.create(
        drivername="postgresql+psycopg",
        username=user,
        password=password,
        host=host,
        port=int(port),
        database=database,
    )

    return create_engine(database_url)