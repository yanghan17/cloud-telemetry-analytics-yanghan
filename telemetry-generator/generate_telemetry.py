"""
generate_telemetry.py

Generates synthetic infrastructure telemetry for a fleet of servers over a
configurable time window, with realistic daily (diurnal) load patterns, a set of
deliberately injected behavioural anomalies, AND a small set of deliberately injected
DATA QUALITY defects.

Design notes (see README / chat writeup for full reasoning):
- Fixed random seed -> reproducible data across runs.
- 5-minute sampling interval -> ~20K rows for 10 servers x 7 days, small
  enough to iterate quickly through Bronze/Silver/Gold/ML stages.
- `is_injected_anomaly` is ground truth used ONLY for evaluating the anomaly
  detection model later. It must never be fed into the model as a feature.

Why data-quality defects, separately from behavioural anomalies
------------------------------------------------------------------
These are two different concepts this project used to conflate by omission. A
*behavioural* anomaly (server01's CPU spike, server03's memory leak, ...) is a real
reading that describes a server actually misbehaving -- Silver should keep it, and the
ML model should flag it. A *data-quality* defect (a null sensor reading, a duplicated
record, a physically impossible CPU of 140%) is telemetry that is simply wrong and
should never reach Gold at all -- Silver's job is to catch and quarantine it.

Before this change, the generator produced perfectly clean data, so Silver's six
reject rules (`silver.py`'s `validity_checks`) had a 0% fire rate on every run --
code that has never executed its main branch is unverified, not correct.
`inject_data_quality_issues` fixes that by appending a small number (40, about 0.2%
of the dataset) of deliberately malformed records, split across every category
Silver is meant to catch. See that function's docstring for exactly how each category
maps to a specific rejection rule, and `silver.py`'s module docstring for a real
ordering bug this exposed and fixed.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone

from telemetry_model import build_servers, generate_baseline, derive_status

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

RANDOM_SEED = 42
NUM_SERVERS = 10
NUM_DAYS = 7
INTERVAL_MINUTES = 5
# Explicitly UTC -- naive datetimes previously got interpreted as the Spark
# session's local timezone on read, shifting every timestamp by the machine offset
# (UTC+8 here → earliest reading landed at 2026-07-12 16:00 UTC). Pinning UTC
# here and `spark.sql.session.timeZone=UTC` in the lakehouse jobs keeps the
# stored instants identical to the generator's intended wall-clock.
START_TIME = datetime(2026, 7, 13, 0, 0, 0, tzinfo=timezone.utc)

OUTPUT_PATH = "../data/telemetry_raw.csv"

rng = np.random.default_rng(RANDOM_SEED)

SERVERS = build_servers(NUM_SERVERS)


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
# Data-quality defect injection
# ----------------------------------------------------------------------

def inject_data_quality_issues(df: pd.DataFrame, rng: np.random.Generator) -> tuple:
    """
    Appends 40 deliberately malformed records (about 0.2% of the dataset) so Silver's
    validation rules have real defects to catch. Every category below is designed to
    net to *zero* change in Silver's valid row count -- see the reasoning per category
    -- so the 20,160-row clean dataset every notebook number and Gold metric is built on
    stays byte-identical; only Bronze's row count and Silver's reject table change.

    Categories, each using a real server_id (a valid foreign key):

    1. exact_duplicate (5 rows) -- a byte-identical copy of an existing reading, same
       (server_id, timestamp). Silver's `dropDuplicates` removes the copy; the original
       survives untouched. Net valid rows: +0.

    2. case_variant_duplicate (5 rows) -- the *same* reading, but hostname re-cased and
       padded with whitespace (e.g. "  WEB-APP-02  "). Tests that Silver's
       standardize-then-dedup ordering collapses this to one row with the canonical
       hostname. (A server_id case-variant, e.g. "SERVER02" vs "server02", is the more
       interesting version of this bug -- it exposed a real ordering issue where Silver
       used to deduplicate *before* standardizing, so the two differently-cased
       readings survived dedup as if they were different servers. That case can't be
       exercised through the CSV -> S3 -> Bronze path on a case-insensitive local
       filesystem: "server_id=SERVER02" and "server_id=server02" collapse to the same
       physical folder on Windows/NTFS or default macOS, silently losing one write --
       a real limitation of testing "the cloud" locally, not a Silver bug. It's covered
       instead by a direct unit test against Silver's transform function --
       see tests/test_silver.py -- which proves the fix without depending on
       filesystem case-sensitivity.) Net valid rows: +0.

    3. missing_value (10 rows, new timestamps) -- one core metric (cpu/memory/disk)
       set to null. Caught by the corresponding `*_out_of_range` rule, all of which
       check `.isNull()` as well as range. Net valid rows: +0, net rejected: +10.

    4. out_of_bounds_timestamp (5 rows, new timestamps) -- the timestamp itself is the
       defect: either before `silver.py`'s MIN_VALID_TIMESTAMP (2019, simulating a
       clock set wrong at provisioning) or far in the future (year 2099, simulating
       clock drift). (A genuinely NULL timestamp was considered too, but ingest.py
       partitions on `timestamp.dt.strftime(...)`, and pandas silently drops NaT
       groups during that groupby -- the row would never reach S3 at all, so Silver
       could never exercise its null-timestamp check this way. That gap in ingest.py
       is a known limitation, not something this generator can route around. A truly
       *ancient* date, e.g. year 1900, was also tried and rejected for a different
       reason: Spark's Parquet writer refuses pre-Gregorian-rebase dates outright
       ["WRITE_ANCIENT_DATETIME"] -- 2019 demonstrates the same MIN_VALID_TIMESTAMP
       check without hitting that unrelated Spark/Parquet limitation.) Net rejected: +5.

    5. cpu_over_100_percent (7 rows, new timestamps) -- a physically impossible sensor
       reading (105-160%). Net rejected: +7.

    6. negative_value (8 rows, new timestamps) -- a negative reading on a metric that
       can never be negative (bytes, IO, error count, latency) -- the signature of a
       corrupted counter or a signed-overflow bug. Net rejected: +8.

    "New timestamps" are the base reading's timestamp offset by a few seconds (never a
    multiple of 300s, so never landing on the real 5-minute grid) -- guarantees these
    rows can't silently overwrite or merge with a genuine reading.
    """
    summary = []
    sample_idx = rng.choice(len(df), size=40, replace=False)
    templates = df.iloc[sample_idx].reset_index(drop=True)

    def offset(ts, seconds):
        return ts + timedelta(seconds=seconds)

    # 1. Exact duplicates.
    exact_dupes = templates.iloc[0:5].copy()
    summary.append(("exact_duplicate", len(exact_dupes)))

    # 2. Case/whitespace-variant duplicates (hostname only -- see docstring for why
    #    server_id casing is tested separately, as a unit test, instead).
    case_variants = templates.iloc[5:10].copy()
    case_variants["hostname"] = "  " + case_variants["hostname"].str.upper() + "  "
    summary.append(("case_variant_duplicate", len(case_variants)))

    # 3. Missing values.
    missing_rows = templates.iloc[10:20].copy().reset_index(drop=True)
    missing_rows["timestamp"] = [offset(ts, 17 + i) for i, ts in enumerate(missing_rows["timestamp"])]
    null_targets = ["cpu_usage_percent", "memory_usage_percent", "disk_usage_percent"] * 4
    for i, col in enumerate(null_targets[: len(missing_rows)]):
        missing_rows.iloc[i, missing_rows.columns.get_loc(col)] = np.nan
    summary.append(("missing_value", len(missing_rows)))

    # 4. Out-of-bounds timestamps (UTC-aware, matching the clean rows).
    bad_ts_rows = templates.iloc[20:25].copy().reset_index(drop=True)
    bad_ts_rows.loc[bad_ts_rows.index[:2], "timestamp"] = pd.Timestamp("2019-01-01", tz="UTC")
    bad_ts_rows.loc[bad_ts_rows.index[2:], "timestamp"] = pd.Timestamp("2099-01-01", tz="UTC")
    summary.append(("out_of_bounds_timestamp", len(bad_ts_rows)))

    # 5. CPU over 100%.
    over_range_rows = templates.iloc[25:32].copy().reset_index(drop=True)
    over_range_rows["timestamp"] = [offset(ts, 31 + i) for i, ts in enumerate(over_range_rows["timestamp"])]
    over_range_rows["cpu_usage_percent"] = rng.uniform(105, 160, len(over_range_rows))
    summary.append(("cpu_over_100_percent", len(over_range_rows)))

    # 6. Negative values.
    negative_rows = templates.iloc[32:40].copy().reset_index(drop=True)
    negative_rows["timestamp"] = [offset(ts, 53 + i) for i, ts in enumerate(negative_rows["timestamp"])]
    negative_targets = (["network_rx_bytes", "disk_io_read", "response_time_ms", "application_error_count"] * 2)
    for i, col in enumerate(negative_targets[: len(negative_rows)]):
        col_idx = negative_rows.columns.get_loc(col)
        negative_rows.iloc[i, col_idx] = -abs(negative_rows.iloc[i, col_idx]) - 1

    # These four categories are synthetic corruption, not real server behaviour --
    # force is_injected_anomaly=0 so they can never be mistaken for a behavioural
    # anomaly during evaluation (the duplicate categories keep whatever value their
    # template row had, since they represent -- deliberately -- the *same* reading).
    for frame in (missing_rows, bad_ts_rows, over_range_rows, negative_rows):
        frame["is_injected_anomaly"] = 0
    summary.append(("negative_value", len(negative_rows)))

    dirty = pd.concat(
        [exact_dupes, case_variants, missing_rows, bad_ts_rows, over_range_rows, negative_rows],
        ignore_index=True,
    )
    return pd.concat([df, dirty], ignore_index=True), summary


# ----------------------------------------------------------------------
# Main generation loop
# ----------------------------------------------------------------------

def main():
    total_points = int((NUM_DAYS * 24 * 60) / INTERVAL_MINUTES)
    timestamps = pd.date_range(start=START_TIME, periods=total_points, freq=f"{INTERVAL_MINUTES}min")

    all_dfs = []
    for server in SERVERS:
        df = generate_baseline(server["server_id"], timestamps, rng)

        # Inject this server's anomaly (if it has one), roughly a third
        # of the way into the week so there's normal data before and after.
        if server["server_id"] in ANOMALY_PLAN:
            start_idx = int(total_points * 0.35)
            df = ANOMALY_PLAN[server["server_id"]](df, start_idx)

        df["hostname"] = server["hostname"]
        all_dfs.append(df)

    result = pd.concat(all_dfs, ignore_index=True)

    # Appended AFTER all baseline/anomaly generation completes, using the same `rng`
    # instance purely for additional draws -- this cannot change a single value in the
    # 20,160 rows generated above; it only adds new rows on top.
    result, dq_summary = inject_data_quality_issues(result, rng)

    result["status"] = result.apply(derive_status, axis=1)

    # Round for readability
    numeric_cols = ["cpu_usage_percent", "memory_usage_percent", "disk_usage_percent",
                     "disk_io_read", "disk_io_write", "network_rx_bytes",
                     "network_tx_bytes", "response_time_ms"]
    result[numeric_cols] = result[numeric_cols].round(2)

    result = result.sort_values(["server_id", "timestamp"]).reset_index(drop=True)

    result.to_csv(OUTPUT_PATH, index=False)
    print(f"Generated {len(result):,} rows across {NUM_SERVERS} servers over {NUM_DAYS} days.")
    print(f"Injected behavioural anomalies on: {list(ANOMALY_PLAN.keys())}")
    print(f"Saved to: {OUTPUT_PATH}")
    print(f"\nAnomaly row counts:\n{result[result['is_injected_anomaly'] == 1]['server_id'].value_counts()}")
    print("\nData-quality defects injected (see inject_data_quality_issues docstring):")
    for category, count in dq_summary:
        print(f"  {category}: {count} rows")


if __name__ == "__main__":
    main()
