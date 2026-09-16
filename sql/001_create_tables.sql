CREATE TABLE IF NOT EXISTS customers (
    customer_id VARCHAR(20) PRIMARY KEY,
    country VARCHAR(50) NOT NULL,
    segment VARCHAR(50) NOT NULL,
    contract_type VARCHAR(50) NOT NULL,
    signup_date DATE NOT NULL,
    monthly_fee NUMERIC(10, 2) NOT NULL,

    transaction_count INTEGER NOT NULL DEFAULT 0,
    total_spend NUMERIC(12, 2) NOT NULL DEFAULT 0,
    avg_transaction_amount NUMERIC(12, 2) NOT NULL DEFAULT 0,
    failed_transactions INTEGER NOT NULL DEFAULT 0,
    refunded_transactions INTEGER NOT NULL DEFAULT 0,
    q2_transactions INTEGER NOT NULL DEFAULT 0,
    q2_failed_transactions INTEGER NOT NULL DEFAULT 0,

    failure_rate DOUBLE PRECISION NOT NULL DEFAULT 0,
    refund_rate DOUBLE PRECISION NOT NULL DEFAULT 0,
    q2_failure_rate DOUBLE PRECISION NOT NULL DEFAULT 0,

    total_tickets INTEGER NOT NULL DEFAULT 0,
    billing_tickets INTEGER NOT NULL DEFAULT 0,
    cancellation_tickets INTEGER NOT NULL DEFAULT 0,
    high_priority_tickets INTEGER NOT NULL DEFAULT 0,
    q2_tickets INTEGER NOT NULL DEFAULT 0,
    q2_billing_tickets INTEGER NOT NULL DEFAULT 0,

    billing_ticket_rate DOUBLE PRECISION NOT NULL DEFAULT 0,
    cancellation_ticket_rate DOUBLE PRECISION NOT NULL DEFAULT 0,
    q2_billing_ticket_rate DOUBLE PRECISION NOT NULL DEFAULT 0,

    churn_probability DOUBLE PRECISION NOT NULL,
    status VARCHAR(20) NOT NULL,

    CONSTRAINT chk_monthly_fee
        CHECK (monthly_fee >= 0),

    CONSTRAINT chk_churn_probability
        CHECK (churn_probability BETWEEN 0 AND 1),

    CONSTRAINT chk_customer_status
        CHECK (status IN ('Active', 'Churned'))
);

CREATE TABLE IF NOT EXISTS transactions (
    transaction_id VARCHAR(20) PRIMARY KEY,
    customer_id VARCHAR(20) NOT NULL,
    transaction_date DATE NOT NULL,
    product VARCHAR(50) NOT NULL,
    channel VARCHAR(50) NOT NULL,
    amount NUMERIC(12, 2) NOT NULL,
    payment_status VARCHAR(20) NOT NULL,

    CONSTRAINT fk_transaction_customer
        FOREIGN KEY (customer_id)
        REFERENCES customers(customer_id),

    CONSTRAINT chk_transaction_amount
        CHECK (amount >= 0),

    CONSTRAINT chk_payment_status
        CHECK (
            payment_status IN (
                'Successful',
                'Failed',
                'Refunded'
            )
        )
);

CREATE TABLE IF NOT EXISTS support_tickets (
    ticket_id VARCHAR(20) PRIMARY KEY,
    customer_id VARCHAR(20) NOT NULL,
    created_at DATE NOT NULL,
    priority VARCHAR(20) NOT NULL,
    category VARCHAR(50) NOT NULL,
    description TEXT NOT NULL,

    CONSTRAINT fk_ticket_customer
        FOREIGN KEY (customer_id)
        REFERENCES customers(customer_id),

    CONSTRAINT chk_ticket_priority
        CHECK (
            priority IN (
                'Low',
                'Medium',
                'High',
                'Critical'
            )
        ),

    CONSTRAINT chk_ticket_category
        CHECK (
            category IN (
                'Technical',
                'Billing',
                'Account',
                'Service',
                'Cancellation'
            )
        )
);