-- Runs once on first Postgres container start (docker-entrypoint-initdb.d).
-- NOT applied by the API's run_migrations(); it only enables pgvector and the
-- migration tracking table. All other schema lives in numbered migrations.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS schema_migrations (
    version VARCHAR(255) PRIMARY KEY,
    applied_at TIMESTAMPTZ DEFAULT now()
);

\echo '000-init completed'
