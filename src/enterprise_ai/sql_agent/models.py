from typing import Any

from pydantic import BaseModel


class SQLQueryResult(BaseModel):
    sql: str
    rows: list[dict[str, Any]]
    truncated: bool = False