"""
load_to_postgres.py

Loads Gold layer data into PostgreSQL as the serving layer for the
dashboard. Deliberately does NOT use Spark/JDBC for this -- see chat
writeup: Gold's output here is tiny (10-70 rows), so reading it via the
lightweight `deltalake` (delta-rs) library straight into Pandas avoids
spinning up a second JVM/Spark session just to move a handful of rows.

Design notes:
- DB credentials are fetched from AWS Secrets Manager at runtime -- never
  hardcoded, consistent with the Terraform setup.
- `servers` and `server_health_summary` are populated now, from Gold.
- `anomalies` is left empty here on purpose: per the target architecture,
  the ML model populates it separately and independently from
  Gold's rule-based signal. Populating it here would blur that boundary.
- `alerts` are derived from health_score thresholds as a simple example of
  turning a Gold metric into an actionable, resolvable operational record.
  For this project, alerts are truncated and reloaded each run (simplest
  correct behavior for a batch demo); a real system would only insert a
  new alert the moment a threshold is first crossed, not on every re-run.

Known limitation: server metadata
(environment, OS, application) is currently hardcoded here to match
generate_telemetry.py's config, rather than flowing through the pipeline
as real data. Propagating it through Bronze/Silver/Gold instead of
duplicating it in two places would be a natural improvement.

Usage:
    export TELEMETRY_DB_SECRET_NAME=cloud-telemetry/db-credentials
    python load_to_postgres.py
"""

import json
import os
import sys

import boto3
import pandas as pd
from deltalake import DeltaTable
from sqlalchemy import create_engine, text

GOLD_SUMMARY_PATH = "../data/lakehouse/gold_server_summary"
SCHEMA_SQL_PATH = "schema.sql"

# Health score -> alert severity thresholds. Illustrative defaults, same
# spirit as the Gold health score weights -- easy to tune, worth being
# ready to justify in review rather than treating as fixed truth.
CRITICAL_HEALTH_THRESHOLD = 60
WARNING_HEALTH_THRESHOLD = 85

# Static server metadata, matching generate_telemetry.py's SERVERS config.
# See module docstring: this duplication is a known limitation.
SERVER_METADATA = {
    f"server{str(i).zfill(2)}": {
        "ip_address": f"10.0.1.{10 + i}",
        "environment": "production",
        "operating_system": "Ubuntu 22.04",
        "application": "order-processing-api",
    }
    for i in range(1, 11)
}


def get_db_credentials() -> dict:
    """Fetch DB connection details from Secrets Manager -- never hardcoded."""
    secret_name = os.environ.get("TELEMETRY_DB_SECRET_NAME", "cloud-telemetry/db-credentials")
    client = boto3.client("secretsmanager")
    try:
        response = client.get_secret_value(SecretId=secret_name)
        return json.loads(response["SecretString"])
    except Exception as e:
        print(f"ERROR: could not fetch secret '{secret_name}' from Secrets Manager: {e}", file=sys.stderr)
        sys.exit(1)


def build_engine(creds: dict):
    url = (
        f"postgresql+psycopg2://{creds['username']}:{creds['password']}"
        f"@{creds['host']}:{creds['port']}/{creds['dbname']}"
    )
    return create_engine(url)


def apply_schema(engine):
    print("Applying schema.sql ...")
    with open(SCHEMA_SQL_PATH) as f:
        schema_sql = f.read()
    with engine.begin() as conn:
        for statement in schema_sql.split(";"):
            statement = statement.strip()
            if statement:
                conn.execute(text(statement))
    print("Schema applied.")


def load_servers(engine, summary_df: pd.DataFrame):
    """Upsert server dimension rows -- ON CONFLICT DO NOTHING keeps this
    safe to re-run without duplicating or clobbering existing rows."""
    print("Loading servers ...")
    with engine.begin() as conn:
        for _, row in summary_df.iterrows():
            meta = SERVER_METADATA.get(row["server_id"], {})
            conn.execute(
                text("""
                    INSERT INTO servers (server_id, hostname, ip_address, environment, operating_system, application)
                    VALUES (:server_id, :hostname, :ip_address, :environment, :operating_system, :application)
                    ON CONFLICT (server_id) DO NOTHING
                """),
                {
                    "server_id": row["server_id"],
                    "hostname": row["hostname"],
                    "ip_address": meta.get("ip_address"),
                    "environment": meta.get("environment"),
                    "operating_system": meta.get("operating_system"),
                    "application": meta.get("application"),
                },
            )
    print(f"  {len(summary_df)} servers upserted.")


def load_health_summary(engine, summary_df: pd.DataFrame):
    """Upsert health summary -- ON CONFLICT DO UPDATE keeps this table as
    a fresh snapshot reflecting the latest pipeline run, per server_id."""
    print("Loading server_health_summary ...")
    cols = [
        "server_id", "avg_cpu_percent", "peak_cpu_percent", "avg_memory_percent",
        "peak_memory_percent", "avg_disk_percent", "peak_disk_percent",
        "avg_network_rx_bytes", "avg_network_tx_bytes", "avg_errors_per_interval",
        "errors_per_hour", "avg_response_time_ms", "peak_response_time_ms",
        "anomaly_count", "total_readings", "disk_trend_percent_change", "health_score",
    ]
    update_cols = [c for c in cols if c != "server_id"]
    set_clause = ", ".join(f"{c} = EXCLUDED.{c}" for c in update_cols)

    with engine.begin() as conn:
        for _, row in summary_df.iterrows():
            params = {c: row[c] for c in cols}
            conn.execute(
                text(f"""
                    INSERT INTO server_health_summary ({', '.join(cols)}, updated_at)
                    VALUES ({', '.join(':' + c for c in cols)}, now())
                    ON CONFLICT (server_id) DO UPDATE SET {set_clause}, updated_at = now()
                """),
                params,
            )
    print(f"  {len(summary_df)} health summary rows upserted.")


def load_alerts(engine, summary_df: pd.DataFrame):
    """
    Derive simple alerts from health_score thresholds. Truncate + reload
    each run -- the simplest CORRECT behavior for a batch demo pipeline
    (see module docstring for why this differs from a real always-on system).
    """
    print("Deriving alerts from health_score thresholds ...")
    alerts = []
    for _, row in summary_df.iterrows():
        if row["health_score"] < CRITICAL_HEALTH_THRESHOLD:
            severity = "CRITICAL"
        elif row["health_score"] < WARNING_HEALTH_THRESHOLD:
            severity = "WARNING"
        else:
            continue
        alerts.append({
            "server_id": row["server_id"],
            "metric": "health_score",
            "severity": severity,
            "message": f"{row['server_id']} health_score is {row['health_score']} "
                       f"(anomaly_count={row['anomaly_count']})",
        })

    with engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE alerts RESTART IDENTITY"))
        for alert in alerts:
            conn.execute(
                text("""
                    INSERT INTO alerts (server_id, metric, severity, message)
                    VALUES (:server_id, :metric, :severity, :message)
                """),
                alert,
            )
    print(f"  {len(alerts)} alerts created ({sum(1 for a in alerts if a['severity']=='CRITICAL')} critical, "
          f"{sum(1 for a in alerts if a['severity']=='WARNING')} warning).")


def main():
    if not os.path.exists(GOLD_SUMMARY_PATH):
        print(f"ERROR: {GOLD_SUMMARY_PATH} not found. Run gold.py first.", file=sys.stderr)
        sys.exit(1)

    print(f"Reading Gold summary from {GOLD_SUMMARY_PATH} (via deltalake, no Spark needed) ...")
    summary_df = DeltaTable(GOLD_SUMMARY_PATH).to_pandas()
    print(f"  {len(summary_df)} rows loaded.")

    creds = get_db_credentials()
    engine = build_engine(creds)

    apply_schema(engine)
    load_servers(engine, summary_df)
    load_health_summary(engine, summary_df)
    load_alerts(engine, summary_df)

    print("\nDone. Verify with:")
    print("  SELECT * FROM server_health_summary ORDER BY health_score ASC;")
    print("  SELECT * FROM alerts;")


if __name__ == "__main__":
    main()
