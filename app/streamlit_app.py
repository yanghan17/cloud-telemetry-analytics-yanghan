"""
streamlit_app.py

Operational dashboard for the telemetry platform. This is the final stop in the target
architecture: Gold -> Postgres -> Streamlit and ML Model -> Postgres -> Streamlit. The
dashboard reads exclusively from PostgreSQL -- never from S3/Delta/Databricks directly
-- so it only needs database credentials to run, independent of where the lakehouse
pipeline happens to be deployed.

Data sources (all in PostgreSQL, see database/schema.sql):
    servers                -- dimension: server_id, hostname, environment, ...
    server_health_summary  -- Gold snapshot: one row per server, refreshed by
                               database/load_to_postgres.py
    anomalies              -- ML detections: one row per incident, refreshed by
                               ml/load_anomalies_to_postgres.py
    alerts                 -- health-score-derived alerts, refreshed by
                               database/load_to_postgres.py

Credentials: TELEMETRY_DATABASE_URL if set (handy for local Postgres / CI), otherwise
fetched from AWS Secrets Manager at runtime -- same convention as every other script in
this project. Never hardcoded.

Usage:
    streamlit run streamlit_app.py
"""

import json
import os

import boto3
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text

# Same thresholds database/load_to_postgres.py uses to derive alerts from health_score --
# duplicated here (rather than imported) because the two scripts run in different
# processes/environments; see the project's existing note on this pattern in
# database/load_to_postgres.py's SERVER_METADATA docstring.
CRITICAL_HEALTH_THRESHOLD = 60
WARNING_HEALTH_THRESHOLD = 85

CACHE_TTL_SECONDS = 60


# ----------------------------------------------------------------------
# Database connection
# ----------------------------------------------------------------------

@st.cache_resource
def get_engine():
    database_url = os.environ.get("TELEMETRY_DATABASE_URL")
    if database_url:
        return create_engine(database_url)

    secret_name = os.environ.get("TELEMETRY_DB_SECRET_NAME", "cloud-telemetry/db-credentials")
    client = boto3.client("secretsmanager")
    creds = json.loads(client.get_secret_value(SecretId=secret_name)["SecretString"])
    url = (
        f"postgresql+psycopg2://{creds['username']}:{creds['password']}"
        f"@{creds['host']}:{creds['port']}/{creds['dbname']}"
    )
    return create_engine(url)


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_health_summary() -> pd.DataFrame:
    query = """
        SELECT s.server_id, s.hostname, s.environment, s.application, h.*
        FROM server_health_summary h
        JOIN servers s USING (server_id)
    """
    df = pd.read_sql(text(query), get_engine())
    return df.loc[:, ~df.columns.duplicated()]


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_anomalies() -> pd.DataFrame:
    return pd.read_sql(text('SELECT * FROM anomalies ORDER BY "timestamp" DESC'), get_engine())


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_alerts() -> pd.DataFrame:
    return pd.read_sql(text("SELECT * FROM alerts ORDER BY triggered_at DESC"), get_engine())


def derive_status(health_score: float) -> str:
    if health_score < CRITICAL_HEALTH_THRESHOLD:
        return "Critical"
    if health_score < WARNING_HEALTH_THRESHOLD:
        return "Warning"
    return "Healthy"


STATUS_ICON = {"Healthy": "\U0001F7E2", "Warning": "\U0001F7E1", "Critical": "\U0001F534"}


# ----------------------------------------------------------------------
# Page
# ----------------------------------------------------------------------

st.set_page_config(page_title="Infrastructure Telemetry Dashboard", layout="wide")

with st.sidebar:
    st.header("Infrastructure Telemetry")
    st.caption("Reads from PostgreSQL only (Gold + ML layers). Data refreshes every "
               f"{CACHE_TTL_SECONDS}s, or click below.")
    if st.button("Refresh now", width='stretch'):
        st.cache_data.clear()
        st.rerun()

try:
    health = load_health_summary()
    anomalies = load_anomalies()
    alerts = load_alerts()
except Exception as exc:
    st.error(
        "Could not connect to PostgreSQL.\n\n"
        f"**Error:** {exc}\n\n"
        "Things to check:\n"
        "- Is the RDS instance running? (`terraform output rds_endpoint`)\n"
        "- Does your current IP match Terraform's `allowed_ip_cidr`? "
        "(`curl checkip.amazonaws.com`, update `terraform.tfvars`, `terraform apply`)\n"
        "- Are AWS credentials configured (`aws sts get-caller-identity`), or is "
        "`TELEMETRY_DATABASE_URL` set for a local/alternate Postgres?\n"
        "- Has `database/load_to_postgres.py` been run at least once to create the schema?"
    )
    st.stop()

health["status"] = health["health_score"].astype(float).apply(derive_status)

with st.sidebar:
    st.divider()
    selected_servers = st.multiselect(
        "Filter servers", options=sorted(health["server_id"].unique()), default=[]
    )
    st.divider()
    st.caption(f"Servers loaded: {len(health)}")
    st.caption(f"Anomaly events: {len(anomalies)}")
    st.caption(f"Alerts: {len(alerts)}")

if selected_servers:
    health_view = health[health["server_id"].isin(selected_servers)]
    anomalies_view = anomalies[anomalies["server_id"].isin(selected_servers)]
    alerts_view = alerts[alerts["server_id"].isin(selected_servers)]
else:
    health_view = health
    anomalies_view = anomalies
    alerts_view = alerts

st.title("Infrastructure Telemetry Dashboard")
st.caption(
    "Serving layer: PostgreSQL, refreshed from the Gold Delta table "
    "(`database/load_to_postgres.py`) and the Isolation Forest anomaly detector "
    "(`ml/anomaly_detection.py` + `ml/load_anomalies_to_postgres.py`)."
)

# ----------------------------------------------------------------------
# KPI row
# ----------------------------------------------------------------------

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Servers", len(health_view))
col2.metric("Healthy", int((health_view["status"] == "Healthy").sum()))
col3.metric("Servers With Alerts", int(alerts_view["server_id"].nunique()))
col4.metric("Detected Anomalies", len(anomalies_view))
col5.metric("Critical Servers", int((health_view["status"] == "Critical").sum()))

st.divider()

# ----------------------------------------------------------------------
# Fleet utilization overview
# ----------------------------------------------------------------------

st.subheader("Fleet Utilization")
metric_cols = st.columns(4)
utilization_metrics = [
    ("CPU (avg)", "avg_cpu_percent", "%"),
    ("Memory (avg)", "avg_memory_percent", "%"),
    ("Disk (avg)", "avg_disk_percent", "%"),
    ("Response time (avg)", "avg_response_time_ms", "ms"),
]
for col, (label, field, unit) in zip(metric_cols, utilization_metrics):
    with col:
        st.caption(label)
        chart_df = health_view.set_index("server_id")[[field]].sort_values(field, ascending=False)
        st.bar_chart(chart_df, height=220)
        fleet_avg = health_view[field].astype(float).mean()
        st.caption(f"Fleet average: {fleet_avg:.1f}{unit}")

st.divider()

# ----------------------------------------------------------------------
# Server health table
# ----------------------------------------------------------------------

st.subheader("Server Health")
display_cols = {
    "server_id": "Server",
    "status": "Status",
    "health_score": "Health Score",
    "avg_cpu_percent": "Avg CPU %",
    "peak_cpu_percent": "Peak CPU %",
    "avg_memory_percent": "Avg Mem %",
    "peak_memory_percent": "Peak Mem %",
    "avg_disk_percent": "Avg Disk %",
    "peak_disk_percent": "Peak Disk %",
    "errors_per_hour": "Errors/hr",
    "avg_response_time_ms": "Avg Resp (ms)",
    "anomaly_count": "Rule Anomalies",
    "disk_trend_percent_change": "Disk Trend (pp)",
}
table = health_view.sort_values("health_score", ascending=True)[list(display_cols)].rename(columns=display_cols)
table["Status"] = table["Status"].map(lambda s: f"{STATUS_ICON[s]} {s}")
st.dataframe(table, width='stretch', hide_index=True)

st.divider()

# ----------------------------------------------------------------------
# Per-server drill-down
# ----------------------------------------------------------------------

st.subheader("Server Detail")
detail_server = st.selectbox("Select a server", sorted(health_view["server_id"].unique()))
detail = health_view[health_view["server_id"] == detail_server].iloc[0]

d1, d2, d3, d4 = st.columns(4)
d1.metric("Status", f"{STATUS_ICON[detail['status']]} {detail['status']}", f"health {detail['health_score']:.0f}/100")
d2.metric("CPU", f"{detail['avg_cpu_percent']:.1f}%", f"peak {detail['peak_cpu_percent']:.1f}%")
d3.metric("Memory", f"{detail['avg_memory_percent']:.1f}%", f"peak {detail['peak_memory_percent']:.1f}%")
d4.metric("Disk", f"{detail['avg_disk_percent']:.1f}%", f"peak {detail['peak_disk_percent']:.1f}%")

d5, d6, d7, d8 = st.columns(4)
d5.metric("Network Rx (avg)", f"{detail['avg_network_rx_bytes'] / 1e6:.2f} MB/5min")
d6.metric("Network Tx (avg)", f"{detail['avg_network_tx_bytes'] / 1e6:.2f} MB/5min")
d7.metric("App Errors", f"{detail['errors_per_hour']:.1f}/hr")
d8.metric("Response Time", f"{detail['avg_response_time_ms']:.0f} ms", f"peak {detail['peak_response_time_ms']:.0f} ms")

server_anomalies = anomalies[anomalies["server_id"] == detail_server]
if not server_anomalies.empty:
    st.caption(f"{len(server_anomalies)} anomaly event(s) detected for {detail_server}:")
    st.dataframe(
        server_anomalies[["timestamp", "metric", "severity", "value", "description"]],
        width='stretch', hide_index=True,
    )
else:
    st.caption(f"No anomalies detected for {detail_server}.")

st.divider()

# ----------------------------------------------------------------------
# Detected anomalies (ML)
# ----------------------------------------------------------------------

st.subheader("Detected Anomalies (Isolation Forest)")
st.caption(
    "One row per detected incident, not per raw reading -- consecutive flagged 5-minute "
    "readings are collapsed into a single event (see `ml/anomaly_detection.py`). "
    "Precision/recall against injected ground truth: 0.66 / 0.69 "
    "(see `notebooks/telemetry_analysis.ipynb` and the model's module docstring)."
)
severity_filter = st.multiselect(
    "Severity", options=["CRITICAL", "WARNING"], default=["CRITICAL", "WARNING"], key="anomaly_severity"
)
anomalies_filtered = anomalies_view[anomalies_view["severity"].isin(severity_filter)]
st.dataframe(
    anomalies_filtered[["server_id", "timestamp", "metric", "value", "severity", "detection_method", "description"]],
    width='stretch', hide_index=True, height=320,
)

st.divider()

# ----------------------------------------------------------------------
# Alerts
# ----------------------------------------------------------------------

st.subheader("Alerts")
unresolved = alerts_view[~alerts_view["resolved"]]
if unresolved.empty:
    st.success("No unresolved alerts.")
else:
    st.dataframe(
        unresolved[["server_id", "triggered_at", "metric", "severity", "message"]],
        width='stretch', hide_index=True,
    )

st.caption(
    "Known limitations: the health score can saturate for servers with a sustained "
    "incident (see notebooks/telemetry_analysis.ipynb), and the ML detector's false "
    "positive rate means a handful of WARNING-severity anomalies on otherwise-healthy "
    "servers are expected -- see the model's module docstring for the full discussion."
)
