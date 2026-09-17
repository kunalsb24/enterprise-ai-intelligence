CREATE INDEX IF NOT EXISTS idx_customers_country_segment
ON customers (country, segment);


CREATE INDEX IF NOT EXISTS idx_transactions_transaction_date
ON transactions (transaction_date);

CREATE INDEX IF NOT EXISTS idx_transactions_customer_id
ON transactions (customer_id);

CREATE INDEX IF NOT EXISTS idx_support_tickets_customer_id
ON support_tickets (customer_id);