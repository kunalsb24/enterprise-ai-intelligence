import pytest

from enterprise_ai.sql_agent.validator import validate_sql


def test_accepts_simple_select():
    validate_sql(
        "SELECT COUNT(*) FROM customers;"
    )


def test_rejects_empty_sql():
    with pytest.raises(
        ValueError,
        match="SQL query cannot be empty",
    ):
        validate_sql("")


def test_rejects_delete():
    with pytest.raises(
        ValueError,
        match="Only SELECT queries are allowed",
    ):
        validate_sql(
            "DELETE FROM customers;"
        )


def test_rejects_drop():
    with pytest.raises(
        ValueError,
        match="Only SELECT queries are allowed",
    ):
        validate_sql(
            "DROP TABLE customers;"
        )


def test_rejects_multiple_statements():
    with pytest.raises(ValueError):
        validate_sql(
            "SELECT * FROM customers; "
            "DROP TABLE customers;"
        )


def test_rejects_invalid_mutating_sql_inside_select():
    with pytest.raises(ValueError):
        validate_sql(
            "SELECT * FROM customers "
            "WHERE customer_id IN "
            "(DELETE FROM customers RETURNING customer_id);"
        )

def test_accepts_approved_table():
    validate_sql(
        "SELECT customer_id, country FROM customers;"
    )


def test_rejects_document_chunks_table():
    with pytest.raises(
        ValueError,
        match="Disallowed table detected",
    ):
        validate_sql(
            "SELECT content FROM document_chunks;"
        )


def test_rejects_unknown_table():
    with pytest.raises(
        ValueError,
        match="Disallowed table detected",
    ):
        validate_sql(
            "SELECT * FROM employee_salaries;"
        )

def test_accepts_join_between_approved_tables():
    sql = """
    SELECT COUNT(*)
    FROM transactions
    JOIN customers
        ON transactions.customer_id = customers.customer_id
    WHERE customers.country = 'Germany'
      AND customers.segment = 'Enterprise'
      AND transactions.payment_status = 'Failed';
    """

    validate_sql(sql)

def test_rejects_join_with_disallowed_table():
    sql = """
    SELECT customers.customer_id
    FROM customers
    JOIN document_chunks
        ON customers.customer_id = document_chunks.document_name;
    """

    with pytest.raises(
        ValueError,
        match="Disallowed table detected",
    ):
        validate_sql(sql)

def test_accepts_subquery_using_approved_tables():
    sql = """
    SELECT customer_id
    FROM customers
    WHERE customer_id IN (
        SELECT customer_id
        FROM transactions
        WHERE payment_status = 'Failed'
    );
    """

    validate_sql(sql)

def test_rejects_disallowed_table_inside_subquery():
    sql = """
    SELECT customer_id
    FROM customers
    WHERE customer_id IN (
        SELECT document_name
        FROM document_chunks
    );
    """

    with pytest.raises(
        ValueError,
        match="Disallowed table detected",
    ):
        validate_sql(sql)

def test_rejects_update_statement():
    sql = """
    UPDATE customers
    SET status = 'Churned'
    WHERE customer_id = 'CUST00001';
    """

    with pytest.raises(
        ValueError,
        match="Only SELECT queries are allowed",
    ):
        validate_sql(sql)

def test_rejects_unknown_column_in_single_table_query():
    sql = """
    SELECT
        c.country,
        c.churned,
        c.total
    FROM customers c;
    """

    with pytest.raises(
        ValueError,
        match="Disallowed column detected",
    ):
        validate_sql(sql)

def test_rejects_hallucinated_churn_columns():
    sql = """
    SELECT
        c.country,
        ROUND(
            c.churned::NUMERIC
            / c.total::NUMERIC
            * 100,
            2
        ) AS churn_rate
    FROM customers c
    WHERE c.status = 'Churned';
    """

    with pytest.raises(
        ValueError,
        match="Disallowed column detected",
    ):
        validate_sql(sql)

def test_accepts_valid_qualified_columns_in_join():
    sql = """
    SELECT
        c.country,
        st.category
    FROM customers c
    JOIN support_tickets st
        ON c.customer_id = st.customer_id
    WHERE c.segment = 'Enterprise';
    """

    validate_sql(sql)


def test_rejects_column_on_wrong_joined_table():
    sql = """
    SELECT
        c.country,
        c.category
    FROM customers c
    JOIN support_tickets st
        ON c.customer_id = st.customer_id;
    """

    with pytest.raises(
        ValueError,
        match="Disallowed column detected",
    ):
        validate_sql(sql)

def test_rejects_unknown_unqualified_column_in_join():
    sql = """
    SELECT
        country,
        made_up_column
    FROM customers c
    JOIN support_tickets st
        ON c.customer_id = st.customer_id;
    """

    with pytest.raises(
        ValueError,
        match="Disallowed column detected",
    ):
        validate_sql(sql)

def test_rejects_ambiguous_unqualified_column_in_join():
    sql = """
    SELECT customer_id
    FROM customers c
    JOIN support_tickets st
        ON c.customer_id = st.customer_id;
    """

    with pytest.raises(
        ValueError,
        match="Ambiguous column",
    ):
        validate_sql(sql)

def test_accepts_order_by_output_alias():
    sql = """
    SELECT
        country,
        COUNT(*) AS customer_count
    FROM customers
    GROUP BY country
    ORDER BY customer_count DESC;
    """

    validate_sql(sql)

def test_accepts_order_by_output_alias_in_join():
    sql = """
    SELECT
        c.country,
        COUNT(st.ticket_id) AS ticket_count
    FROM customers c
    JOIN support_tickets st
        ON c.customer_id = st.customer_id
    GROUP BY c.country
    ORDER BY ticket_count DESC;
    """

    validate_sql(sql)