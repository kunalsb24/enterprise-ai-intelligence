-- Create the restricted role used by the SQL Analytics Agent.
--
-- The role itself is created separately because its login password
-- must come from environment configuration, not source control.
--
-- This migration configures the permissions that the role receives.

GRANT CONNECT ON DATABASE enterprise_ai
TO enterprise_ai_reader;

GRANT USAGE ON SCHEMA public
TO enterprise_ai_reader;

GRANT SELECT ON
    customers,
    transactions,
    support_tickets
TO enterprise_ai_reader;

REVOKE ALL ON document_chunks
FROM enterprise_ai_reader;