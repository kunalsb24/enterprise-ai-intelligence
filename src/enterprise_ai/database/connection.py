import os

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine


load_dotenv()


def get_database_url() -> str:
    """Build the PostgreSQL connection URL from environment variables."""

    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5433")
    database = os.getenv("POSTGRES_DB", "enterprise_ai")
    user = os.getenv("POSTGRES_USER", "enterprise_user")
    password = os.getenv("POSTGRES_PASSWORD", "enterprise_password")

    return (
        f"postgresql+psycopg://{user}:{password}"
        f"@{host}:{port}/{database}"
    )


def get_engine() -> Engine:
    """Create a SQLAlchemy engine for PostgreSQL."""

    return create_engine(get_database_url())