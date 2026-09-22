SQL_SCHEMA_CONTEXT = """
You have access to the following PostgreSQL business tables.

Table: customers
Purpose: One row per customer, including profile information,
aggregated behavioral features, churn probability, and customer status.

Important columns:
- customer_id: unique customer identifier
- country: customer country
- segment: customer segment
- contract_type: customer contract type
- signup_date: customer signup date
- monthly_fee: monthly customer fee
- q2_transactions: number of transactions during Q2
- q2_failed_transactions: failed transactions during Q2
- q2_failure_rate: transaction failure rate during Q2
- q2_tickets: support tickets during Q2
- q2_billing_tickets: billing support tickets during Q2
- q2_billing_ticket_rate: billing-ticket rate during Q2
- churn_probability: modeled churn probability
- status: either Active or Churned

Important semantic distinctions:
- segment describes the customer business segment. Values include Enterprise.
- contract_type describes the customer's contract arrangement and is different from segment.
- q2_failure_rate is the customer's transaction failure rate during Q2 2025.
- q2_failed_transactions is the customer's count of failed transactions during Q2 2025.
- q2_transactions is the customer's total transaction count during Q2 2025.
- If a question asks for the average customer-level Q2 failure rate, use AVG(q2_failure_rate).
- A rate is stored as a decimal fraction, for example 0.20 means 20%.
- Observed customer churn rate for a group is calculated from the status
  column: the number of customers whose status is 'Churned' divided by
  the total number of customers in that group.
- Do not invent churned or total columns. They do not exist.
- Do not use churn_probability when the question asks for the observed
  churn rate.
- When calculating observed churn rate, do not filter the WHERE clause
  to only Churned customers, because the denominator must include all
  customers in the group.
- Use conditional aggregation to count Churned customers while keeping
  all customers available for the denominator.

Table: transactions
Purpose: One row per customer transaction.

Columns:
- transaction_id: unique transaction identifier
- customer_id: references customers.customer_id
- transaction_date: transaction date
- product: purchased product
- channel: transaction channel
- amount: transaction amount
- payment_status: Successful, Failed, or Refunded

Table: support_tickets
Purpose: One row per customer support ticket.

Columns:
- ticket_id: unique ticket identifier
- customer_id: references customers.customer_id
- created_at: ticket creation date
- priority: Low, Medium, High, or Critical
- category: Technical, Billing, Account, Service, or Cancellation
- description: ticket description

Data context:
- All q2_* columns in the customers table refer to Q2 2025
  (April 1 through June 30, 2025).
- status is the observed customer outcome: Active or Churned.
- churn_probability is a modeled probability and is not the same as
  the observed status.
- Prefer the aggregated customers columns when they directly answer
  the business question.
- Use transactions or support_tickets when the question requires
  transaction-level, ticket-level, date-specific, product-specific,
  channel-specific, priority-specific, or category-specific analysis.

Table usage guidance:
- customers has one row per customer. Use it for customer counts, customer attributes, status, and aggregated customer-level features.
- transactions has one row per transaction. Use it when the question asks to count, filter, or analyze individual transactions.
- support_tickets has one row per support ticket. Use it when the question asks to count, filter, or analyze individual support tickets.
- category and created_at belong to support_tickets, not customers.
- payment_status and transaction_date belong to transactions, not customers.
- country and segment belong to customers.
- When a question requires attributes from customers plus individual transactions or support tickets, join the tables using customer_id.
- For questions about events during Q2 2025, use a half-open date range on the relevant event table: date >= '2025-04-01' AND date < '2025-07-01'.

Relationships:
- transactions.customer_id references customers.customer_id
- support_tickets.customer_id references customers.customer_id
"""