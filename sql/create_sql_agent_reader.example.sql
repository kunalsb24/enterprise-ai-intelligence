-- One-time local setup for the SQL Analytics Agent database role.
--
-- IMPORTANT:
-- Replace the placeholder below with a strong local password before
-- running this command. Never commit a real database password to Git.
--
-- The permissions for this role are configured separately by:
-- sql/004_configure_sql_agent_reader.sql

CREATE ROLE enterprise_ai_reader
WITH
    LOGIN
    NOSUPERUSER
    NOCREATEDB
    NOCREATEROLE
    NOREPLICATION
    PASSWORD '<YOUR_LOCAL_PASSWORD>';