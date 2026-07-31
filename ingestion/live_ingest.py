"""
live_ingest.py

Continuously generates and uploads telemetry, on an interval, for as long as the
process runs. This is what makes the assessment brief's requirement literally true:
"the important requirement is that telemetry can continuously enter the platform."

Why this is a separate script rather than a `--continuous` flag on ingest.py or
generate_telemetry.py
------------------------------------------------------------------------------------
generate_telemetry.py + ingest.py build ONE fixed, reproducible, seeded 7-day dataset.
Every number quoted in notebooks/telemetry_analysis.ipynb, the Gold health scores, and
the ML model's precision/recall are all computed against that exact dataset -- if
either script grew a mode that changed what it writes on every run, re-running the
"batch" path would silently invalidate every one of those numbers. Continuous
ingestion needs to exist alongside that fixed dataset, not replace or perturb it, so
it lives in its own script with its own (deliberately unseeded) random generator.

What it actually does
----------------------
Every `--interval-seconds` (default 300, matching the 5-minute sampling interval used
everywhere else in this project), it:
  1. Generates one fresh reading per server for "right now", using the *exact same*
     `generate_baseline` / diurnal-cycle model as the batch generator (imported from
     telemetry_model.py -- see that module's docstring for why sharing the model
     matters: a live reading and a historical reading of the same server at the same
     time of day should look statistically identical, and only will if both paths
     call the same code).
  2. Writes one small Parquet file per server, into the SAME Hive-style partition
     layout ingest.py uses (server_id=X/dt=YYYY-MM-DD/), but with a unique,
     timestamped filename per tick rather than ingest.py's fixed "telemetry.parquet" --
     ingest.py's batch write is a "here is the whole day, replace it" operation;
     continuous ingestion is "here is one more reading, add it" and must never
     overwrite the ticks already written earlier that day.
  3. Sleeps, then repeats -- indefinitely, or `--max-ticks` times for a bounded demo.

Bronze picks these up with zero code changes: bronze.py already reads
`{RAW_INPUT_PATH}/server_id=*/dt=*/`, a directory glob that includes every Parquet
file in each partition folder, not just one -- it was written for the batch case but
happens to generalise to "many small files accumulating over the day" for free.

In a real deployment this loop would be a scheduled job (a systemd timer, a cron
entry, an EventBridge-triggered Lambda, an Airflow DAG) rather than a long-running
Python process kept alive in a terminal -- this script models the payload of that job
(one tick, uploaded) so the *logic* is identical either way; only the thing that
invokes it on a schedule would change.

Usage:
    # Dry run: local folder instead of S3, one tick every 10 seconds, stop after 5 ticks
    python live_ingest.py --dry-run --interval-seconds 10 --max-ticks 5

    # Real, indefinite: uploads to S3 every 5 minutes until stopped with Ctrl+C
    export TELEMETRY_S3_BUCKET=your-bucket-name
    python live_ingest.py
"""

import argparse
import io
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# telemetry_model.py lives in a sibling directory (telemetry-generator/), not a
# separate installed package -- this repo doesn't ship a setup.py/pyproject.toml, so a
# small sys.path addition is the simplest way to share it without restructuring every
# other script's imports.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "telemetry-generator"))
from telemetry_model import build_servers, generate_baseline, derive_status  # noqa: E402

try:
    import boto3
    from botocore.exceptions import NoCredentialsError, ClientError
except ImportError:
    boto3 = None

NUM_SERVERS = 10
DEFAULT_INTERVAL_SECONDS = 300  # matches the 5-minute sampling interval used elsewhere
SERVERS = build_servers(NUM_SERVERS)


def generate_tick(rng: np.random.Generator, now: datetime) -> pd.DataFrame:
    """One fresh reading per server, for the current instant. Deliberately does not
    call any of generate_telemetry.py's anomaly injectors -- those model the six fixed
    historical incidents the ML model is evaluated against; a live tick is meant to
    represent ordinary telemetry continuously arriving, not a scripted incident."""
    # `now` is always timezone-aware UTC from run_one_tick(); keep that on the index
    # so Parquet/Spark don't re-interpret a naive wall-clock in a local session TZ.
    ts_index = pd.DatetimeIndex([pd.Timestamp(now).tz_convert("UTC")])
    rows = []
    for server in SERVERS:
        row = generate_baseline(server["server_id"], ts_index, rng)
        row["hostname"] = server["hostname"]
        rows.append(row)
    tick_df = pd.concat(rows, ignore_index=True)
    tick_df["status"] = tick_df.apply(derive_status, axis=1)
    return tick_df


def to_parquet_bytes(df: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    df.to_parquet(buffer, engine="pyarrow", index=False)
    return buffer.getvalue()


def tick_filename() -> str:
    # HHMMSS + a short random suffix: unique within a day even if two ticks somehow
    # land in the same second, without needing any shared counter/state between runs.
    return f"telemetry_{datetime.now(timezone.utc):%H%M%S}_{uuid.uuid4().hex[:6]}.parquet"


def upload_dry_run(server_id: str, dt: str, body: bytes, local_root: str) -> None:
    filename = tick_filename()
    key_path = Path(local_root) / f"server_id={server_id}" / f"dt={dt}" / filename
    key_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.write_bytes(body)
    print(f"[dry-run] wrote {key_path} ({len(body):,} bytes)")


def upload_s3(server_id: str, dt: str, body: bytes, bucket: str, prefix: str) -> None:
    if boto3 is None:
        print("ERROR: boto3 is not installed. Run: pip install boto3", file=sys.stderr)
        sys.exit(1)

    s3 = boto3.client("s3")  # credentials picked up from env / AWS CLI config
    filename = tick_filename()
    key = f"{prefix}/server_id={server_id}/dt={dt}/{filename}"
    try:
        s3.put_object(Bucket=bucket, Key=key, Body=body)
        print(f"[s3] uploaded s3://{bucket}/{key} ({len(body):,} bytes)")
    except NoCredentialsError:
        print("ERROR: No AWS credentials found. Configure via `aws configure` "
              "or export AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY.", file=sys.stderr)
        sys.exit(1)
    except ClientError as e:
        print(f"ERROR uploading {key}: {e}", file=sys.stderr)
        sys.exit(1)


def run_one_tick(rng: np.random.Generator, args) -> int:
    now = datetime.now(timezone.utc)
    tick_df = generate_tick(rng, now)
    dt = now.strftime("%Y-%m-%d")

    for server_id, group in tick_df.groupby("server_id"):
        body = to_parquet_bytes(group.drop(columns=["dt"], errors="ignore"))
        if args.dry_run:
            upload_dry_run(server_id, dt, body, args.local_root)
        else:
            upload_s3(server_id, dt, body, args.bucket, args.prefix)

    return len(tick_df)


def main():
    parser = argparse.ArgumentParser(description="Continuously generate and ingest telemetry.")
    parser.add_argument("--dry-run", action="store_true", help="Write to local folder instead of S3")
    parser.add_argument("--local-root", default="../data/s3-mirror/raw", help="Local output root for dry-run mode")
    parser.add_argument("--prefix", default="raw", help="S3 key prefix (folder) for uploaded objects")
    parser.add_argument("--interval-seconds", type=float, default=DEFAULT_INTERVAL_SECONDS,
                         help="Seconds between ticks (default: 300, i.e. 5 minutes)")
    parser.add_argument("--max-ticks", type=int, default=None,
                         help="Stop after N ticks (omit to run until Ctrl+C, as a real scheduled job would)")
    args = parser.parse_args()

    args.bucket = os.environ.get("TELEMETRY_S3_BUCKET")
    if not args.dry_run and not args.bucket:
        print("ERROR: TELEMETRY_S3_BUCKET environment variable is not set.\n"
              "  export TELEMETRY_S3_BUCKET=your-bucket-name   (Git Bash)\n"
              "  $env:TELEMETRY_S3_BUCKET='your-bucket-name'   (PowerShell)\n"
              "Or pass --dry-run to test locally without AWS.", file=sys.stderr)
        sys.exit(1)

    rng = np.random.default_rng()  # unseeded on purpose -- see module docstring
    tick_count = 0
    target = f"bucket={args.bucket}" if not args.dry_run else f"local folder={args.local_root}"
    print(f"Starting continuous ingestion ({target}, interval={args.interval_seconds}s). "
          f"Press Ctrl+C to stop.")

    try:
        while args.max_ticks is None or tick_count < args.max_ticks:
            now_str = datetime.now(timezone.utc).isoformat(timespec="seconds")
            server_count = run_one_tick(rng, args)
            tick_count += 1
            print(f"Tick {tick_count} complete at {now_str} ({server_count} servers).")
            if args.max_ticks is None or tick_count < args.max_ticks:
                time.sleep(args.interval_seconds)
    except KeyboardInterrupt:
        print("\nStopped by user.")

    print(f"\nDone. Ingested {tick_count} tick(s) "
          f"({tick_count * NUM_SERVERS} total telemetry records this run).")
    print("Run bronze.py next to pick up the new partition files -- its glob read "
          "already covers every file under each server_id=*/dt=*/ folder.")


if __name__ == "__main__":
    main()
