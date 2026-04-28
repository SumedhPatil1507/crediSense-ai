-- CrediSense AI — Supabase Schema
-- Run this in the Supabase SQL Editor (supabase.com -> your project -> SQL Editor)

-- Predictions table
CREATE TABLE IF NOT EXISTS predictions (
    id               TEXT PRIMARY KEY,
    timestamp        TIMESTAMPTZ DEFAULT NOW(),
    income_lpa       NUMERIC(10,2),
    age_years        INTEGER,
    experience_years INTEGER,
    income_norm      NUMERIC(6,4),
    age_norm         NUMERIC(6,4),
    experience_norm  NUMERIC(6,4),
    risk_prob        NUMERIC(6,4),
    ci_lower         NUMERIC(6,4),
    ci_upper         NUMERIC(6,4),
    decision         TEXT,
    confidence       TEXT,
    page             TEXT DEFAULT 'API'
);

-- Feedback table
CREATE TABLE IF NOT EXISTS feedback (
    id              TEXT PRIMARY KEY,
    timestamp       TIMESTAMPTZ DEFAULT NOW(),
    prediction_id   TEXT REFERENCES predictions(id) ON DELETE SET NULL,
    feedback        TEXT CHECK (feedback IN ('correct','incorrect','unsure')),
    corrected_label TEXT,
    notes           TEXT
);

-- Audit log table
CREATE TABLE IF NOT EXISTS audit_log (
    id          TEXT PRIMARY KEY,
    timestamp   TIMESTAMPTZ DEFAULT NOW(),
    event       TEXT,
    input_hash  TEXT,
    user_id     TEXT,
    details     TEXT
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON predictions(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_predictions_decision  ON predictions(decision);
CREATE INDEX IF NOT EXISTS idx_feedback_prediction   ON feedback(prediction_id);
CREATE INDEX IF NOT EXISTS idx_audit_event           ON audit_log(event);

-- Row Level Security (enable for multi-tenant)
ALTER TABLE predictions ENABLE ROW LEVEL SECURITY;
ALTER TABLE feedback     ENABLE ROW LEVEL SECURITY;
ALTER TABLE audit_log    ENABLE ROW LEVEL SECURITY;

-- Allow service role full access (used by backend)
CREATE POLICY "service_role_all" ON predictions FOR ALL USING (true);
CREATE POLICY "service_role_all" ON feedback     FOR ALL USING (true);
CREATE POLICY "service_role_all" ON audit_log    FOR ALL USING (true);
