# Database (PostgreSQL + pgvector)

Numbered SQL migrations. `000-init.sql` runs once on first container start
(Postgres init mount) and enables `pgvector` + the `schema_migrations` table.
All other `NNN-*.sql` files are applied by the API on startup via
`db.run_migrations()`, in numeric order, each tracked in `schema_migrations`.

## Conventions

- Name files `NNN-short-name.sql`.
- Start each (non-init) migration with:
  ```sql
  INSERT INTO schema_migrations (version) VALUES ('NNN-short-name')
      ON CONFLICT (version) DO NOTHING;
  ```
- Use `CREATE TABLE IF NOT EXISTS`, `ALTER TABLE ... IF NOT EXISTS`, etc. so
  migrations are re-runnable.
- Never edit an applied migration; add a new numbered file instead.

## Tables

| Table | Purpose |
|-------|---------|
| `knowledge_base` | RAG corpus with `vector(768)` embeddings (nomic-embed-text) |
| `diagnoses` | Structured LLM diagnoses |
| `work_orders` | Draft maintenance work orders from the agent |
| `llm_calls` | AI observability (latency, tokens, retrieval quality) |
| `eval_runs` | Eval harness results |

The image is `pgvector/pgvector:pg16` (Postgres 16 with pgvector preinstalled).
