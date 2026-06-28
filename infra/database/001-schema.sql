-- Foreshock schema: RAG knowledge base (pgvector), diagnoses + work orders,
-- and observability/eval tables. Idempotent and re-runnable.

INSERT INTO schema_migrations (version) VALUES ('001-schema')
    ON CONFLICT (version) DO NOTHING;

CREATE EXTENSION IF NOT EXISTS vector;

-- Retrieval-augmented knowledge base. embedding dim = nomic-embed-text (768).
CREATE TABLE IF NOT EXISTS knowledge_base (
    id SERIAL PRIMARY KEY,
    fault_type VARCHAR(32) NOT NULL,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    source TEXT,
    embedding vector(768),
    created_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_kb_fault ON knowledge_base(fault_type);
-- The corpus is small (dozens of rows), so an exact cosine scan is instant; no
-- ANN (ivfflat/hnsw) index is needed.

-- Structured LLM diagnoses.
CREATE TABLE IF NOT EXISTS diagnoses (
    id SERIAL PRIMARY KEY,
    sample_id TEXT,
    predicted_condition VARCHAR(32),
    confidence DOUBLE PRECISION,
    severity VARCHAR(16),
    summary TEXT,
    recommended_actions JSONB,
    sources JSONB,
    model TEXT,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Draft work orders emitted by the agent.
CREATE TABLE IF NOT EXISTS work_orders (
    id SERIAL PRIMARY KEY,
    diagnosis_id INTEGER REFERENCES diagnoses(id) ON DELETE SET NULL,
    asset TEXT,
    condition VARCHAR(32),
    priority VARCHAR(16),
    actions JSONB,
    status VARCHAR(16) DEFAULT 'draft',
    created_at TIMESTAMPTZ DEFAULT now()
);

-- AI observability: one row per LLM/embedding call.
CREATE TABLE IF NOT EXISTS llm_calls (
    id SERIAL PRIMARY KEY,
    operation VARCHAR(32),
    model TEXT,
    latency_ms DOUBLE PRECISION,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    total_tokens INTEGER,
    retrieval_score DOUBLE PRECISION,
    ok BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_llm_calls_created ON llm_calls(created_at DESC);

-- Eval harness results.
CREATE TABLE IF NOT EXISTS eval_runs (
    id SERIAL PRIMARY KEY,
    suite VARCHAR(64),
    total INTEGER,
    passed INTEGER,
    diagnosis_accuracy DOUBLE PRECISION,
    retrieval_precision DOUBLE PRECISION,
    retrieval_recall DOUBLE PRECISION,
    hallucination_rate DOUBLE PRECISION,
    details JSONB,
    created_at TIMESTAMPTZ DEFAULT now()
);
