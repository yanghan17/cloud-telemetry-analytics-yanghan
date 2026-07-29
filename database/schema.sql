-- schema.sql
--
-- PostgreSQL schema for the telemetry serving layer.
--
-- Design notes (see chat writeup for full reasoning):
-- - `servers` consolidates the assessment's suggested "servers" +
--   "server_inventory" tables -- they described the same single dimension,
--   splitting them would be a pointless 1:1 join with no benefit.
-- - `server_health_summary` is a serving-layer materialization of the
--   gold_server_summary Delta table, refreshed on each pipeline run.
-- - `anomalies` (immutable historical detection events) and `alerts`
--   (actionable, has a resolved/unresolved lifecycle) are kept separate
--   because they represent genuinely different kinds of things, not just
--   a naming preference.

CREATE TABLE IF NOT EXISTS servers (
    server_id           VARCHAR(20) PRIMARY KEY,
    hostname            VARCHAR(100) NOT NULL,
    ip_address          VARCHAR(45),              -- supports IPv4 and IPv6
    environment         VARCHAR(50) NOT NULL DEFAULT 'production',
    operating_system    VARCHAR(100),
    application         VARCHAR(100),
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Serving-layer copy of gold_server_summary. Overwritten wholesale on each
-- pipeline run (see load_to_postgres.py) rather than incrementally updated,
-- since it's a small, fully-recomputed snapshot each time -- simpler and
-- safer than reconciling partial updates for a table this size.
CREATE TABLE IF NOT EXISTS server_health_summary (
    server_id                   VARCHAR(20) PRIMARY KEY REFERENCES servers(server_id),
    avg_cpu_percent             NUMERIC(5,2),
    peak_cpu_percent            NUMERIC(5,2),
    avg_memory_percent          NUMERIC(5,2),
    peak_memory_percent         NUMERIC(5,2),
    avg_disk_percent            NUMERIC(5,2),
    peak_disk_percent           NUMERIC(5,2),
    avg_network_rx_bytes        NUMERIC(14,2),
    avg_network_tx_bytes        NUMERIC(14,2),
    avg_errors_per_interval     NUMERIC(6,2),
    errors_per_hour             NUMERIC(6,2),
    avg_response_time_ms        NUMERIC(8,2),
    peak_response_time_ms       NUMERIC(8,2),
    anomaly_count               INTEGER NOT NULL DEFAULT 0,
    total_readings              INTEGER,
    disk_trend_percent_change   NUMERIC(6,2),
    health_score                NUMERIC(5,1),
    updated_at                  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Immutable historical record of a specific detected anomaly event.
-- detection_method distinguishes rule-based (available now) from ML-based
-- (added once the Day 8 anomaly detection model exists) -- see chat writeup
-- on why Gold's rule-based anomaly_count and the ML model are kept as
-- separate, independently explainable signals.
CREATE TABLE IF NOT EXISTS anomalies (
    anomaly_id      SERIAL PRIMARY KEY,
    server_id       VARCHAR(20) NOT NULL REFERENCES servers(server_id),
    "timestamp"     TIMESTAMPTZ NOT NULL,
    metric          VARCHAR(50) NOT NULL,
    value           NUMERIC(14,4),
    severity        VARCHAR(20) NOT NULL,     -- WARNING | CRITICAL
    detection_method VARCHAR(30) NOT NULL,    -- rule_based | ml_isolation_forest
    description     TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_anomalies_server_id ON anomalies(server_id);
CREATE INDEX IF NOT EXISTS idx_anomalies_timestamp ON anomalies("timestamp");

-- Actionable, operational entity with a resolution lifecycle -- distinct
-- from `anomalies` (see design note above).
CREATE TABLE IF NOT EXISTS alerts (
    alert_id        SERIAL PRIMARY KEY,
    server_id       VARCHAR(20) NOT NULL REFERENCES servers(server_id),
    triggered_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    metric          VARCHAR(50) NOT NULL,
    severity        VARCHAR(20) NOT NULL,     -- WARNING | CRITICAL
    message         TEXT,
    resolved        BOOLEAN NOT NULL DEFAULT FALSE,
    resolved_at     TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_alerts_server_id_resolved ON alerts(server_id, resolved);
