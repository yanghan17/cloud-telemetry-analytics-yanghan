"""
load_anomalies_to_postgres.py

Loads the ML anomaly events (produced by anomaly_detection.py) into the `anomalies`
table in PostgreSQL. This is deliberately a separate script from
database/load_to_postgres.py, mirroring the two independent arrows into Postgres in
the target architecture diagram: Gold -> Postgres (servers, server_health_summary,
alerts) and ML Model -> Postgres (anomalies). Keeping them separate means the ML step
can be re-run and reloaded on its own schedule without touching the Gold snapshot, and
vice versa.

Design notes (see database/load_to_postgres.py for the matching Gold-side loader):
- DB credentials come from AWS Secrets Manager at runtime -- never hardcoded, same
  convention as the rest of this project.
- `anomalies` is truncated and reloaded each run, the same "simplest correct" choice
  load_to_postgres.py makes for `alerts`: this project re-runs the whole pipeline over
  the same fixed week of synthetic data rather than continuously appending, so
  reloading is safer than trying to reconcile partial updates.
- Only rows for servers that already exist in the `servers` table will insert cleanly
  (anomalies.server_id is a foreign key) -- run database/load_to_postgres.py first.

Usage:
    export TELEMETRY_DB_SECRET_NAME=cloud-telemetry/db-credentials
    python load_anomalies_to_postgres.py
"""

import json
import os
import sys

import boto3
import pandas as pd
from sqlalchemy import create_engine, text

EVENTS_PATH = os.environ.get("TELEMETRY_ML_EVENTS_OUTPUT", "../data/ml/anomaly_events.parquet")


def get_db_credentials() -> dict:
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


def load_anomalies(engine, events_df: pd.DataFrame):
    print("Loading anomalies (truncate + reload, same convention as alerts) ...")
    cols = ["server_id", "timestamp", "metric", "value", "severity", "detection_method", "description"]
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE anomalies RESTART IDENTITY"))
        for _, row in events_df.iterrows():
            conn.execute(
                text("""
                    INSERT INTO anomalies (server_id, "timestamp", metric, value, severity, detection_method, description)
                    VALUES (:server_id, :timestamp, :metric, :value, :severity, :detection_method, :description)
                """),
                {c: row[c] for c in cols},
            )
    print(f"  {len(events_df)} anomalies loaded "
          f"({(events_df['severity'] == 'CRITICAL').sum()} critical, "
          f"{(events_df['severity'] == 'WARNING').sum()} warning).")


def main():
    if not os.path.exists(EVENTS_PATH):
        print(f"ERROR: {EVENTS_PATH} not found. Run anomaly_detection.py first.", file=sys.stderr)
        sys.exit(1)

    print(f"Reading anomaly events from {EVENTS_PATH} ...")
    events_df = pd.read_parquet(EVENTS_PATH)
    print(f"  {len(events_df)} events loaded.")

    creds = get_db_credentials()
    engine = build_engine(creds)
    load_anomalies(engine, events_df)

    print("\nDone. Verify with:")
    print("  SELECT server_id, COUNT(*) FROM anomalies GROUP BY server_id ORDER BY COUNT(*) DESC;")


if __name__ == "__main__":
    main()
