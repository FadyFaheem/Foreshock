-- Add asset + rms to diagnoses so the agent can trend health per asset over time.

INSERT INTO schema_migrations (version) VALUES ('002-diagnosis-trend')
    ON CONFLICT (version) DO NOTHING;

ALTER TABLE diagnoses ADD COLUMN IF NOT EXISTS asset VARCHAR(64);
ALTER TABLE diagnoses ADD COLUMN IF NOT EXISTS rms DOUBLE PRECISION;
CREATE INDEX IF NOT EXISTS idx_diagnoses_asset ON diagnoses(asset, created_at DESC);
