"""
generate_telemetry.py

Generates synthetic infrastructure telemetry for a fleet of servers over a
configurable time window, with realistic daily (diurnal) load patterns and
a set of deliberately injected anomalies.

Design notes (see README / chat writeup for full reasoning):
- Fixed random seed -> reproducible data across runs.
- 5-minute sampling interval -> ~20K rows for 10 servers x 7 days, small
  enough to iterate quickly through Bronze/Silver/Gold/ML stages.
- `is_injected_anomaly` is ground truth used ONLY for evaluating the anomaly
  detection model later. It must never be fed into the model as a feature.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

RANDOM_SEED = 42
NUM_SERVERS = 10
NUM_DAYS = 7
INTERVAL_MINUTES = 5
START_TIME = datetime(2026, 7, 13, 0, 0, 0)  # a Monday, 7 days before "now"

OUTPUT_PATH = "../data/telemetry_raw.csv"

rng = np.random.default_rng(RANDOM_SEED)

SERVERS = [
    {"server_id": f"server{str(i).zfill(2)}", "hostname": f"web-app-{str(i).zfill(2)}",
     "environment": "production", "operating_system": "Ubuntu 22.04",
     "application": "order-processing-api"}
    for i in range(1, NUM_SERVERS + 1)
]


# ----------------------------------------------------------------------
# Baseline signal generation
# ----------------------------------------------------------------------

def diurnal_multiplier(timestamps: pd.DatetimeIndex) -> np.ndarray:
    """
    Returns a multiplier (roughly 0.6 - 1.3) that peaks during business
    hours (9am-6pm) and dips overnight, to give CPU/network a realistic
    daily rhythm instead of flat random noise.
    """
    hours = timestamps.hour + timestamps.minute / 60.0
    # Shift so the peak of the sine wave lands around 1pm (13:00)
    radians = 2 * np.pi * (hours - 7) / 24
    wave = np.sin(radians)
    return 0.95 + 0.35 * wave  # ranges ~0.6 to ~1.3


def generate_baseline(server_id: str, timestamps: pd.DatetimeIndex) -> pd.DataFrame:
    """Generate normal (non-anomalous) telemetry for one server."""
    n = len(timestamps)
    daily = diurnal_multiplier(timestamps)

    cpu = np.clip(30 * daily + rng.normal(0, 4, n), 2, 100)
    memory = np.clip(45 + rng.normal(0, 3, n), 5, 100)
    disk = np.clip(50 + rng.normal(0, 1, n), 5, 100)
    disk_io_read = np.clip(rng.normal(120, 20, n) * daily, 0, None)
    disk_io_write = np.clip(rng.normal(80, 15, n) * daily, 0, None)
    net_rx = np.clip(rng.normal(500_000, 80_000, n) * daily, 0, None)
    net_tx = np.clip(rng.normal(300_000, 60_000, n) * daily, 0, None)
    errors = rng.poisson(0.5, n)  # occasional background errors, near-zero baseline
    response_time = np.clip(rng.normal(120, 15, n), 20, None)

    return pd.DataFrame({
        "timestamp": timestamps,
        "server_id": server_id,
        "cpu_usage_percent": cpu,
        "memory_usage_percent": memory,
        "disk_usage_percent": disk,
        "disk_io_read": disk_io_read,
        "disk_io_write": disk_io_write,
        "network_rx_bytes": net_rx,
        "network_tx_bytes": net_tx,
        "application_error_count": errors,
        "response_time_ms": response_time,
        "is_injected_anomaly": 0,
    })


# ----------------------------------------------------------------------
# Anomaly injectors
# One function per anomaly type, matching the assessment's suggested scenarios.
# Each takes a server's DataFrame and mutates a specific time window.
# ----------------------------------------------------------------------

def inject_cpu_spike(df, start_idx, duration_rows=24):
    """server01: CPU suddenly jumps to 95-100% for a sustained burst."""
    end_idx = min(start_idx + duration_rows, len(df))
    df.loc[start_idx:end_idx, "cpu_usage_percent"] = rng.uniform(95, 100, end_idx - start_idx + 1)
    df.loc[start_idx:end_idx, "is_injected_anomaly"] = 1
    return df


def inject_memory_leak(df, start_idx, duration_rows=288):
    """server03: memory climbs gradually over ~24h (288 rows at 5-min interval)."""
    end_idx = min(start_idx + duration_rows, len(df))
    n = end_idx - start_idx + 1
    ramp = np.linspace(0, 45, n)  # gradually adds up to 45 extra percentage points
    df.loc[start_idx:end_idx, "memory_usage_percent"] = np.clip(
        df.loc[start_idx:end_idx, "memory_usage_percent"].values + ramp, 0, 100
    )
    df.loc[start_idx:end_idx, "is_injected_anomaly"] = 1
    return df


def inject_disk_fill(df, start_idx, duration_rows=576):
    """server05: disk usage gradually climbs toward ~95% over ~48h."""
    end_idx = min(start_idx + duration_rows, len(df))
    n = end_idx - start_idx + 1
    ramp = np.linspace(0, 40, n)
    df.loc[start_idx:end_idx, "disk_usage_percent"] = np.clip(
        df.loc[start_idx:end_idx, "disk_usage_percent"].values + ramp, 0, 100
    )
    df.loc[start_idx:end_idx, "is_injected_anomaly"] = 1
    return df


def inject_network_surge(df, start_idx, duration_rows=12):
    """server07: sudden, sharp spike in network traffic (e.g. DDoS-like burst)."""
    end_idx = min(start_idx + duration_rows, len(df))
    df.loc[start_idx:end_idx, "network_rx_bytes"] *= rng.uniform(8, 12)
    df.loc[start_idx:end_idx, "network_tx_bytes"] *= rng.uniform(8, 12)
    df.loc[start_idx:end_idx, "is_injected_anomaly"] = 1
    return df


def inject_error_spike(df, start_idx, duration_rows=18):
    """server08: application error count suddenly spikes (e.g. bad deploy)."""
    end_idx = min(start_idx + duration_rows, len(df))
    df.loc[start_idx:end_idx, "application_error_count"] = rng.integers(20, 60, end_idx - start_idx + 1)
    df.loc[start_idx:end_idx, "is_injected_anomaly"] = 1
    return df


def inject_slow_response(df, start_idx, duration_rows=36):
    """server10: response time degrades significantly (e.g. DB contention)."""
    end_idx = min(start_idx + duration_rows, len(df))
    df.loc[start_idx:end_idx, "response_time_ms"] *= rng.uniform(6, 10)
    df.loc[start_idx:end_idx, "is_injected_anomaly"] = 1
    return df


# Map each anomaly to the server it belongs to, per the assessment brief.
ANOMALY_PLAN = {
    "server01": inject_cpu_spike,
    "server03": inject_memory_leak,
    "server05": inject_disk_fill,
    "server07": inject_network_surge,
    "server08": inject_error_spike,
    "server10": inject_slow_response,
}


# ----------------------------------------------------------------------
# Main generation loop
# ----------------------------------------------------------------------

def derive_status(row):
    """Simple rule-based status label, independent of the ML model."""
    if row["cpu_usage_percent"] > 90 or row["memory_usage_percent"] > 90 or row["disk_usage_percent"] > 90:
        return "CRITICAL"
    if row["cpu_usage_percent"] > 75 or row["memory_usage_percent"] > 75 or row["application_error_count"] > 10:
        return "WARNING"
    return "OK"


def main():
    total_points = int((NUM_DAYS * 24 * 60) / INTERVAL_MINUTES)
    timestamps = pd.date_range(start=START_TIME, periods=total_points, freq=f"{INTERVAL_MINUTES}min")

    all_dfs = []
    for server in SERVERS:
        df = generate_baseline(server["server_id"], timestamps)

        # Inject this server's anomaly (if it has one), roughly a third
        # of the way into the week so there's normal data before and after.
        if server["server_id"] in ANOMALY_PLAN:
            start_idx = int(total_points * 0.35)
            df = ANOMALY_PLAN[server["server_id"]](df, start_idx)

        df["hostname"] = server["hostname"]
        all_dfs.append(df)

    result = pd.concat(all_dfs, ignore_index=True)
    result["status"] = result.apply(derive_status, axis=1)

    # Round for readability
    numeric_cols = ["cpu_usage_percent", "memory_usage_percent", "disk_usage_percent",
                     "disk_io_read", "disk_io_write", "network_rx_bytes",
                     "network_tx_bytes", "response_time_ms"]
    result[numeric_cols] = result[numeric_cols].round(2)

    result = result.sort_values(["server_id", "timestamp"]).reset_index(drop=True)

    result.to_csv(OUTPUT_PATH, index=False)
    print(f"Generated {len(result):,} rows across {NUM_SERVERS} servers over {NUM_DAYS} days.")
    print(f"Injected anomalies on: {list(ANOMALY_PLAN.keys())}")
    print(f"Saved to: {OUTPUT_PATH}")
    print(f"\nAnomaly row counts:\n{result[result['is_injected_anomaly'] == 1]['server_id'].value_counts()}")


if __name__ == "__main__":
    main()
