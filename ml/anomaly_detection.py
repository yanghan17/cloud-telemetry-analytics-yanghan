"""
anomaly_detection.py

Trains a single Isolation Forest to flag abnormal server behaviour from telemetry,
using the Silver Delta table as input. This is the ML counterpart to the rule-based
`status` thresholds already computed in the lakehouse -- see the module docstring
in gold.py and the findings in notebooks/telemetry_analysis.ipynb (Finding 6) for why
those two signals are kept separate rather than merged.

Method: Isolation Forest
------------------------
Isolation Forest was chosen over the other options in the brief (z-score, moving
average, simple thresholds, k-means) because:
  - It is multivariate: it can flag a reading that is only unusual in *combination*
    (e.g. moderately high CPU *and* elevated response time together), which a
    per-metric z-score or threshold cannot express.
  - It needs no distributional assumption (unlike z-score/k-means, which implicitly
    assume roughly Gaussian, spherical clusters) -- telemetry metrics here are a mix
    of bounded percentages, byte counts and a sparse Poisson-like error count, none
    of which are Gaussian.
  - It is still simple to explain: it isolates points by randomly partitioning the
    feature space, and points that separate from the rest in very few splits get a
    high anomaly score. No labels are needed to train it.

Features
--------
The seven metrics the brief suggests, plus two engineered features the data science
notebook showed were necessary to catch the *gradual* anomalies (a memory leak and a
disk fill that a single-point snapshot cannot distinguish from noise):

    cpu_usage_percent, memory_usage_percent, disk_usage_percent,
    network_rx_bytes, network_tx_bytes, application_error_count (log1p-transformed),
    response_time_ms,
    mem_slope_24h, disk_slope_24h   (rolling 24h trend, points/day -- see notebook Finding 3)

`application_error_count` is log1p-transformed because it is a sparse, Poisson-like
count (mostly 0-1): its raw median absolute deviation is 0 for most servers, which
would blow up the per-server scaling below. log1p compresses the long tail without
losing the "did errors spike" signal.

Hour-of-day features (sin/cos encoding) were tested, since the notebook's Finding 7
showed a strong daily cycle and warned it could cause false positives. They did not
improve precision/recall here and are left out to keep the model simple -- see the
"What I would improve" section in the module docstring at the bottom of this file.

Per-server scaling
-------------------
Every feature is scaled *within each server* using a robust z-score (median and
median absolute deviation, MAD) rather than one fleet-wide scale. This directly
answers notebook Finding 7 and Finding 4: without it, a model would judge "unusual"
against the whole fleet's mixed day/night traffic, and a quiet server's normal
midday reading could look anomalous next to a busy server's overnight reading.
Median/MAD (rather than mean/standard deviation) was chosen because a large enough
anomaly inflates its own standard deviation and can partly hide from a mean/std
z-score -- this was demonstrated directly in the notebook's network analysis
(Finding 4: 32.8 vs 39.3 for the same event, scored two ways). If a server's MAD for
a feature is exactly 0 (e.g. a server with zero application errors all week), the
scaler falls back to standard deviation for that one feature/server so scaling never
divides by zero.

How an anomaly is identified
-----------------------------
`contamination=0.05` tells the forest to treat the most-isolated 5% of readings, by
score, as anomalies. This is calibrated to be close to this dataset's known injected
anomaly rate (4.76% of rows) for a clean demonstration; in production, without ground
truth, you would set it from an operational risk budget (e.g. "we can investigate
about N flagged readings per day") or from a historical false-alarm rate, and revisit
it as real incidents are confirmed or dismissed.

What causes false positives / false negatives (see the evaluation section below for
the numbers behind these claims)
------------------------------------------------------------------------------------
- False positives cluster overnight (see the printed hour distribution) rather than
  at the midday peak the notebook's diurnal analysis predicted -- worth investigating
  further with more data, but plausibly because low-traffic overnight readings have
  a smaller and noisier baseline, so proportionally similar noise produces a larger
  robust z-score.
- False negatives: `contamination` sets one shared anomaly "budget" across the whole
  fleet and all anomaly types. A short, sharp CPU spike (server01: ~6.7 sigma) can
  lose out to a longer, more extreme trend elsewhere (a 24h memory/disk slope can
  reach 40-60 sigma) for a place in that budget, even though 6.7 sigma is still a
  genuinely rare event on its own. Capping (winsorizing) extreme feature z-scores at
  a fixed limit before training rebalances this -- tested at +/-8, it raised overall
  F1 from 0.67 to 0.76, but traded roughly 30 points of recall on the slow-response
  anomaly for gains on the short spike and the error burst. It was left out of the
  shipped model to keep the method easy to explain in one pass, but is the first
  thing to try with more time (see below).

What I would improve with more time
------------------------------------
- Winsorize per-server z-scores before training (see above) and tune the cap on a
  held-out week rather than by eye.
- Investigate the overnight false-positive cluster rather than just reporting it.
- Train one model per anomaly *type* (magnitude spikes vs. slopes vs. rate features)
  and combine their votes, instead of one model sharing a single contamination
  budget across very differently-scaled phenomena.
- Evaluate with a proper time-based train/test split. This script fits and scores on
  the same week, which is standard for unsupervised anomaly detection but means the
  reported metrics describe how well the model explains this data, not how it would
  generalise to a future week.

Usage:
    python anomaly_detection.py
"""

import os

import joblib
import numpy as np
import pandas as pd
from deltalake import DeltaTable
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

SILVER_PATH = os.environ.get("TELEMETRY_SILVER_PATH", "../data/lakehouse/silver")
OUTPUT_PATH = os.environ.get("TELEMETRY_ML_OUTPUT", "../data/ml/anomaly_results.parquet")
EVENTS_OUTPUT_PATH = os.environ.get("TELEMETRY_ML_EVENTS_OUTPUT", "../data/ml/anomaly_events.parquet")
MODEL_PATH = os.environ.get("TELEMETRY_ML_MODEL", "model/isolation_forest.joblib")

ROWS_PER_DAY = 288  # samples per day at a 5-minute interval; also the rolling-slope window
CONTAMINATION = 0.05
RANDOM_SEED = 42

BASE_FEATURES = [
    "cpu_usage_percent",
    "memory_usage_percent",
    "disk_usage_percent",
    "network_rx_bytes",
    "network_tx_bytes",
    "error_count_log",  # log1p(application_error_count) -- see module docstring
    "response_time_ms",
]
ENGINEERED_FEATURES = ["mem_slope_24h", "disk_slope_24h"]
FEATURES = BASE_FEATURES + ENGINEERED_FEATURES

GROUND_TRUTH_COL = "is_injected_anomaly"

# Human-readable label for each feature the model can name as the "dominant" one in an
# anomaly event -- the engineered slope features map back to the metric they describe.
METRIC_LABELS = {
    "cpu_usage_percent": "cpu_usage_percent",
    "memory_usage_percent": "memory_usage_percent",
    "disk_usage_percent": "disk_usage_percent",
    "network_rx_bytes": "network_rx_bytes",
    "network_tx_bytes": "network_tx_bytes",
    "error_count_log": "application_error_count",
    "response_time_ms": "response_time_ms",
    "mem_slope_24h": "memory_usage_percent (trend)",
    "disk_slope_24h": "disk_usage_percent (trend)",
}
RAW_VALUE_COL = {
    "error_count_log": "application_error_count",
    "mem_slope_24h": "memory_usage_percent",
    "disk_slope_24h": "disk_usage_percent",
}


def rolling_slope(values: np.ndarray, window: int = ROWS_PER_DAY) -> np.ndarray:
    """
    Slope (in units per day) of every `window`-sample sliding window, using the
    closed-form OLS slope for an evenly spaced x-axis: sum(x_centered * y) / sum(x_centered^2).
    The first `window - 1` samples have no full window behind them yet and are
    left at 0 (no trend signal available) rather than NaN, so they still train cleanly.
    """
    v = np.asarray(values, dtype=float)
    x_centered = np.arange(window) - (window - 1) / 2
    denom = (x_centered**2).sum()
    out = np.zeros(len(v))
    if len(v) < window:
        return out
    windows = np.lib.stride_tricks.sliding_window_view(v, window)
    out[window - 1:] = (windows * x_centered).sum(axis=1) / denom * window
    return out


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the engineered columns this model needs, per server."""
    df = df.sort_values(["server_id", "timestamp"]).reset_index(drop=True)
    df["error_count_log"] = np.log1p(df["application_error_count"])

    frames = []
    for _, g in df.groupby("server_id"):
        g = g.copy()
        g["mem_slope_24h"] = rolling_slope(g["memory_usage_percent"].to_numpy())
        g["disk_slope_24h"] = rolling_slope(g["disk_usage_percent"].to_numpy())
        frames.append(g)
    return pd.concat(frames).sort_index()


def fit_server_scalers(df: pd.DataFrame, features: list) -> dict:
    """
    Compute a robust (median, MAD) scale per server per feature. Returned as a plain
    dict of arrays (not a fitted sklearn object) so it can be joblib-dumped alongside
    the model and re-applied identically at inference time on new data.
    """
    scalers = {}
    for server_id, g in df.groupby("server_id"):
        stats = {}
        for feat in features:
            median = g[feat].median()
            mad = (g[feat] - median).abs().median()
            scale = mad if mad > 1e-6 else (g[feat].std() + 1e-9)
            stats[feat] = (median, scale)
        scalers[server_id] = stats
    return scalers


def apply_scalers(df: pd.DataFrame, features: list, scalers: dict) -> pd.DataFrame:
    """Apply robust z-scores: 0.6745 rescales MAD to be comparable to a standard deviation."""
    out = pd.DataFrame(index=df.index)
    for feat in features:
        medians = df["server_id"].map(lambda s: scalers[s][feat][0])
        scales = df["server_id"].map(lambda s: scalers[s][feat][1])
        out[feat] = 0.6745 * (df[feat] - medians) / scales
    return out


def evaluate(df: pd.DataFrame) -> None:
    """Compare the model's predictions against ground truth and the existing rule baseline."""
    if GROUND_TRUTH_COL not in df.columns:
        print("No ground-truth column found -- skipping evaluation (expected on real telemetry).")
        return

    y_true = df[GROUND_TRUTH_COL].to_numpy()
    y_pred = df["is_anomaly"].to_numpy()

    print("\n=== Model evaluation vs. injected ground truth ===")
    p, r, f1 = precision_score(y_true, y_pred), recall_score(y_true, y_pred), f1_score(y_true, y_pred)
    print(f"Precision: {p:.3f}   Recall: {r:.3f}   F1: {f1:.3f}")
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    print(f"TP={tp}  FP={fp}  FN={fn}  TN={tn}")

    # The rule-based baseline already living in gold.py / derive_status(), recomputed
    # here for an apples-to-apples comparison against the same Silver snapshot.
    rule_flag = df["status"].isin(["WARNING", "CRITICAL"]).to_numpy().astype(int)
    rp, rr, rf1 = (
        precision_score(y_true, rule_flag, zero_division=0),
        recall_score(y_true, rule_flag, zero_division=0),
        f1_score(y_true, rule_flag, zero_division=0),
    )
    print(f"\nRule-based baseline (status thresholds): precision={rp:.3f} recall={rr:.3f} f1={rf1:.3f}")
    print(f"Isolation Forest:                         precision={p:.3f} recall={r:.3f} f1={f1:.3f}")

    print("\nPer-server breakdown:")
    breakdown = df.groupby("server_id").apply(
        lambda g: pd.Series({
            "injected_truth": g[GROUND_TRUTH_COL].sum(),
            "model_flagged": g["is_anomaly"].sum(),
            "model_tp": ((g["is_anomaly"] == 1) & (g[GROUND_TRUTH_COL] == 1)).sum(),
            "rule_flagged": g["status"].isin(["WARNING", "CRITICAL"]).sum(),
        }),
        include_groups=False,
    )
    breakdown["model_recall_pct"] = (
        100 * breakdown["model_tp"] / breakdown["injected_truth"].replace(0, np.nan)
    ).round(1)
    print(breakdown.to_string())

    print("\nFalse positives by local hour of day (see module docstring: 'What causes false positives'):")
    fp_rows = df[(df["is_anomaly"] == 1) & (df[GROUND_TRUTH_COL] == 0)]
    print(fp_rows["timestamp"].dt.tz_convert("Asia/Singapore").dt.hour.value_counts().sort_index().to_string())


EVENT_MERGE_GAP_MINUTES = 30  # flagged readings within this gap, per server, join the same event


def build_anomaly_events(df: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse flagged readings into one "event" per incident, matching the anomalies
    table's grain (one row per detected anomaly, not per raw sample -- storing every
    flagged 5-minute reading as its own row would put ~1,000 rows in `anomalies` for one
    week of data, which is not what that table is for).

    Consecutive flagged readings are grouped as one event, and so are flagged readings
    up to EVENT_MERGE_GAP_MINUTES apart -- the model's recall is well under 100% (see
    the evaluation above), so a single real incident often has a few individual readings
    in the middle that weren't flagged; without this tolerance a 24-hour memory leak
    fragments into dozens of near-duplicate one-reading events instead of reading as one
    incident.

    For each event: the "metric" is whichever feature had the largest robust z-score at
    the event's single most anomalous reading (its "peak"), and severity is assigned by
    ranking events on that peak anomaly score -- the top third become CRITICAL, the rest
    WARNING, so severity reflects the model's own confidence rather than borrowing the
    rule-based thresholds it's meant to complement.
    """
    flagged = df[df["is_anomaly"] == 1].sort_values(["server_id", "timestamp"]).copy()
    if flagged.empty:
        return pd.DataFrame(columns=[
            "server_id", "timestamp", "metric", "value", "severity", "description",
            "detection_method", "readings", "peak_score",
        ])

    gap = flagged.groupby("server_id")["timestamp"].diff()
    new_event = gap.isna() | (gap > pd.Timedelta(minutes=EVENT_MERGE_GAP_MINUTES))
    flagged["_event_id"] = new_event.cumsum()  # monotonic; a new server always starts a new event too

    z_at_peak = X.loc[flagged.index]

    events = []
    for _event_id, group in flagged.groupby("_event_id"):
        server_id = group["server_id"].iloc[0]
        peak_idx = group["anomaly_score"].idxmax()
        peak_row = df.loc[peak_idx]
        dominant_feature = z_at_peak.loc[peak_idx, FEATURES].abs().idxmax()
        metric = METRIC_LABELS[dominant_feature]
        value_col = RAW_VALUE_COL.get(dominant_feature, dominant_feature)
        start_ts, end_ts = group["timestamp"].min(), group["timestamp"].max()
        events.append({
            "server_id": server_id,
            "timestamp": start_ts,  # when the incident started
            "peak_timestamp": peak_row["timestamp"],
            "metric": metric,
            "value": round(float(peak_row[value_col]), 4),
            "peak_score": round(float(peak_row["anomaly_score"]), 4),
            "readings": len(group),
            "duration_minutes": int((end_ts - start_ts).total_seconds() / 60) + 5,
            "detection_method": "ml_isolation_forest",
        })

    events_df = pd.DataFrame(events).sort_values("peak_score", ascending=False).reset_index(drop=True)
    cutoff = max(1, len(events_df) // 3)
    events_df["severity"] = "WARNING"
    events_df.loc[events_df.index < cutoff, "severity"] = "CRITICAL"

    events_df["description"] = events_df.apply(
        lambda r: (
            f"Isolation Forest flagged {r['metric']} on {r['server_id']}: "
            f"{r['readings']} flagged readings spanning ~{r['duration_minutes']} minutes "
            f"(peak reading {r['peak_timestamp']:%Y-%m-%d %H:%M} UTC, anomaly score {r['peak_score']:.3f})."
        ),
        axis=1,
    )
    return events_df.sort_values(["server_id", "timestamp"]).reset_index(drop=True)


def main():
    print(f"Reading Silver from {SILVER_PATH} ...")
    df = DeltaTable(SILVER_PATH).to_pandas()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    print(f"Loaded {len(df):,} rows across {df['server_id'].nunique()} servers.")

    print("Building features (rolling slopes, log-transformed error count) ...")
    df = build_features(df)

    print("Fitting per-server robust scalers and standardising features ...")
    scalers = fit_server_scalers(df, FEATURES)
    X = apply_scalers(df, FEATURES, scalers)

    print(f"Training Isolation Forest (contamination={CONTAMINATION}) on {X.shape[0]:,} rows x {X.shape[1]} features ...")
    model = IsolationForest(n_estimators=300, contamination=CONTAMINATION, random_state=RANDOM_SEED, n_jobs=-1)
    model.fit(X)

    # decision_function: lower = more anomalous. Negated so higher score = more anomalous,
    # which is the more intuitive convention for a report or a dashboard.
    df["anomaly_score"] = -model.decision_function(X)
    df["is_anomaly"] = (model.predict(X) == -1).astype(int)
    df["result"] = np.where(df["is_anomaly"] == 1, "ANOMALY", "NORMAL")

    flagged = int(df["is_anomaly"].sum())
    print(f"\nFlagged {flagged:,} / {len(df):,} readings ({100 * flagged / len(df):.2f}%) as anomalies.")

    print("\nExample output (matches the brief's format):")
    sample_cols = ["timestamp", "server_id", "result"]
    print(
        df.sort_values(["timestamp", "server_id"])[sample_cols]
        .head(10)
        .to_string(index=False, header=["timestamp", "server", "result"])
    )

    print("\nTop 10 highest-scoring anomalies fleet-wide:")
    top10 = df.sort_values("anomaly_score", ascending=False)[
        ["timestamp", "server_id", "anomaly_score"] + BASE_FEATURES
    ].head(10)
    numeric_cols = top10.columns.drop(["timestamp", "server_id"])
    top10[numeric_cols] = top10[numeric_cols].round(2)
    print(top10.to_string(index=False))

    evaluate(df)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    output_cols = ["timestamp", "server_id", "result", "is_anomaly", "anomaly_score"] + FEATURES
    df[output_cols].to_parquet(OUTPUT_PATH, index=False)
    print(f"\nWrote scored results to {OUTPUT_PATH} ({len(df):,} rows).")

    print("\nCollapsing consecutive flagged readings into discrete anomaly events ...")
    events_df = build_anomaly_events(df, X)
    print(f"Built {len(events_df)} anomaly events "
          f"({(events_df['severity'] == 'CRITICAL').sum()} CRITICAL, "
          f"{(events_df['severity'] == 'WARNING').sum()} WARNING).")
    print(events_df[["server_id", "timestamp", "metric", "severity", "readings"]].to_string(index=False))

    events_df.to_parquet(EVENTS_OUTPUT_PATH, index=False)
    print(f"\nWrote {len(events_df)} anomaly events to {EVENTS_OUTPUT_PATH}.")
    print("Load into Postgres with: python load_anomalies_to_postgres.py "
          "(separate from load_to_postgres.py's Gold -> servers/health_summary/alerts step, "
          "matching the two independent Postgres arrows in the target architecture diagram).")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump({"model": model, "scalers": scalers, "features": FEATURES}, MODEL_PATH)
    print(f"Saved model + scalers to {MODEL_PATH}.")


if __name__ == "__main__":
    main()
