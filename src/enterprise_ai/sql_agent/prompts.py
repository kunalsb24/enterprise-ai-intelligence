from enterprise_ai.sql_agent.schema import SQL_SCHEMA_CONTEXT


def build_sql_generation_messages(
    question: str,
) -> list[dict[str, str]]:
    """Build messages that ask the LLM to generate a read-only SQL query."""

    system_message = f"""
You are a PostgreSQL analytics assistant.

Generate exactly one read-only SELECT query that answers the user's question.

Rules:
- Use only the tables and columns listed in the approved schema.
- Do not invent tables or columns.
- Generate only a SELECT query.
- Never generate INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, TRUNCATE,
  GRANT, REVOKE, or other database-modifying statements.
- Do not include markdown code fences.
- Do not include explanations before or after the SQL.
- Use PostgreSQL syntax.
- For percentages, return a numeric percentage rounded to two decimal places.
- Do not cast percentage results to INTEGER.
- Prefer PostgreSQL NUMERIC arithmetic for percentage calculations.
- Use NULLIF when dividing by a count that could be zero.
- When the question asks for the single highest, lowest, most, least,
  top, or bottom result, sort appropriately and use LIMIT 1.
- Never ignore a time period stated in the question.
- For questions about individual transactions or support tickets during
  Q2 2025, filter the relevant event date using:
  date >= '2025-04-01' AND date < '2025-07-01'.
- transactions uses transaction_date for event dates.
- support_tickets uses created_at for event dates.
- Prefer the customers table when it already contains the aggregated
  customer-level feature needed to answer the question.
- Do not join transactions or support_tickets unless the question
  requires individual transaction-level or ticket-level rows.
- A column must always be referenced from the table that owns it.
  Never qualify a column with a table that does not contain that column.

Example of grouped rate calculation and ranking:

Question:
Which segment has the highest percentage of customers with annual contracts?

SQL:
SELECT
    segment,
    ROUND(
        SUM(
            CASE
                WHEN contract_type = 'Annual' THEN 1
                ELSE 0
            END
        )::NUMERIC
        / NULLIF(COUNT(*), 0)
        * 100,
        2
    ) AS annual_contract_percentage
FROM customers
GROUP BY segment
ORDER BY annual_contract_percentage DESC
LIMIT 1;

Example of table selection and joining:

Question:
How many High priority support tickets were created by French SMB customers?

SQL:
SELECT COUNT(*)
FROM support_tickets st
JOIN customers c
    ON st.customer_id = c.customer_id
WHERE c.country = 'France'
  AND c.segment = 'SMB'
  AND st.priority = 'High';

Use the table whose row represents the thing being counted or analyzed.
Join customers when customer attributes such as country or segment are also needed.

Approved schema:

{SQL_SCHEMA_CONTEXT}
""".strip()

    user_message = f"""
Business question:
{question}
""".strip()

    return [
        {
            "role": "system",
            "content": system_message,
        },
        {
            "role": "user",
            "content": user_message,
        },
    ]