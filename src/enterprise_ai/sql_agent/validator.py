import re

import sqlglot
from sqlglot import exp


FORBIDDEN_SQL_KEYWORDS = {
    "INSERT",
    "UPDATE",
    "DELETE",
    "DROP",
    "ALTER",
    "CREATE",
    "TRUNCATE",
    "GRANT",
    "REVOKE",
}


ALLOWED_TABLES = {
    "customers",
    "transactions",
    "support_tickets",
}

ALLOWED_COLUMNS = {
    "customers": {
        "customer_id",
        "country",
        "segment",
        "contract_type",
        "signup_date",
        "monthly_fee",
        "transaction_count",
        "total_spend",
        "avg_transaction_amount",
        "failed_transactions",
        "refunded_transactions",
        "q2_transactions",
        "q2_failed_transactions",
        "failure_rate",
        "refund_rate",
        "q2_failure_rate",
        "total_tickets",
        "billing_tickets",
        "cancellation_tickets",
        "high_priority_tickets",
        "q2_tickets",
        "q2_billing_tickets",
        "billing_ticket_rate",
        "cancellation_ticket_rate",
        "q2_billing_ticket_rate",
        "churn_probability",
        "status",
    },
    "transactions": {
        "transaction_id",
        "customer_id",
        "transaction_date",
        "product",
        "channel",
        "amount",
        "payment_status",
    },
    "support_tickets": {
        "ticket_id",
        "customer_id",
        "created_at",
        "priority",
        "category",
        "description",
    },
}


def validate_sql(sql: str) -> None:
    """Validate that generated SQL is a single read-only SELECT query."""

    normalized_sql = sql.strip()

    if not normalized_sql:
        raise ValueError("SQL query cannot be empty.")

    sql_without_trailing_semicolon = normalized_sql.rstrip(";")

    if ";" in sql_without_trailing_semicolon:
        raise ValueError("Multiple SQL statements are not allowed.")

    try:
        parsed_query = sqlglot.parse_one(
            normalized_sql,
            read="postgres",
        )
    except sqlglot.errors.ParseError as exc:
        raise ValueError("Invalid SQL syntax.") from exc

    if not isinstance(parsed_query, exp.Select):
        raise ValueError("Only SELECT queries are allowed.")

    words = set(
        re.findall(
            r"\b[A-Z]+\b",
            normalized_sql.upper(),
        )
    )

    forbidden_found = words.intersection(
        FORBIDDEN_SQL_KEYWORDS
    )

    if forbidden_found:
        raise ValueError(
            "Forbidden SQL keyword detected: "
            + ", ".join(sorted(forbidden_found))
        )

    referenced_tables = {
        table.name.lower()
        for table in parsed_query.find_all(exp.Table)
    }

    disallowed_tables = referenced_tables - ALLOWED_TABLES

    if disallowed_tables:
        raise ValueError(
            "Disallowed table detected: "
            + ", ".join(sorted(disallowed_tables))
        )

    if len(referenced_tables) == 1:
        table_name = next(iter(referenced_tables))
        allowed_columns = ALLOWED_COLUMNS[table_name]

        referenced_columns = {
            column.name.lower()
            for column in parsed_query.find_all(exp.Column)
        }

        select_aliases = {
            expression.alias.lower()
            for expression in parsed_query.expressions
            if expression.alias
        }

        disallowed_columns = (
            referenced_columns
            - allowed_columns
            - select_aliases
        )

        if disallowed_columns:
            raise ValueError(
                "Disallowed column detected: "
                + ", ".join(
                    sorted(disallowed_columns)
                )
            )

    if parsed_query.args.get("joins"):
        alias_to_table = {}

        select_aliases = {
            expression.alias.lower()
            for expression in parsed_query.expressions
            if expression.alias
        }

        for table in parsed_query.find_all(exp.Table):
            table_name = table.name.lower()
            alias_name = table.alias_or_name.lower()

            alias_to_table[alias_name] = table_name

        for column in parsed_query.find_all(exp.Column):
            column_name = column.name.lower()
            table_alias = column.table.lower()

            if not table_alias:
                if column_name in select_aliases:
                    continue
                
                matching_tables = [
                    table_name
                    for table_name in referenced_tables
                    if column_name in ALLOWED_COLUMNS[table_name]
                ]

                if not matching_tables:
                    raise ValueError(
                        "Disallowed column detected: "
                        f"{column_name}"
                    )

                if len(matching_tables) > 1:
                    raise ValueError(
                        "Ambiguous column detected: "
                        f"{column_name}"
                    )

                continue

            table_name = alias_to_table.get(table_alias)

            if table_name is None:
                raise ValueError(
                    f"Unknown table alias: {table_alias}"
                )

            allowed_columns = ALLOWED_COLUMNS[
                table_name
            ]

            if column_name not in allowed_columns:
                raise ValueError(
                    "Disallowed column detected: "
                    f"{table_alias}.{column_name}"
                )



    