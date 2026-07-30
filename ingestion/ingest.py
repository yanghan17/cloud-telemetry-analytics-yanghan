"""
ingest.py

Reads the locally generated telemetry CSV and "ingests" it into cloud
object storage (S3), one Parquet file per server per day, using Hive-style
partitioning (server_id=X/dt=YYYY-MM-DD/) so downstream Spark/Databricks
jobs can do partition pruning.

Design notes (see chat writeup for full reasoning):
- Converts CSV rows -> Parquet (columnar, binary, compressed) per partition.
  Parquet is used here (rather than JSON Lines) so the ingestion layer
  itself produces a format Spark/Delta can read natively and efficiently,
  with schema/types preserved -- no re-inferring types downstream.
- Bucket name and AWS credentials come from the environment ONLY.
  Never hardcode secrets here.
- Supports --dry-run to write to a local folder instead of real S3,
  useful before your AWS account/credentials are ready.
- `server_id` is dropped from each partition file before writing, same as `dt` --
  both are already encoded in the Hive-style partition folder name
  (server_id=X/dt=Y/), and Bronze re-derives them from that folder name via
  partition discovery. Storing them again inside the file is redundant, and
  was previously causing Spark to warn `COLUMN_ALREADY_EXISTS` on read.
- Each partition file gets a unique, content-independent filename rather than a
  fixed "telemetry.parquet" -- a fixed name means a second write to the same
  partition silently overwrites the first, which is exactly what happened when
  testing this pipeline with data containing inconsistent server_id casing:
  "server_id=SERVER02" and "server_id=server02" resolve to the SAME physical
  folder on a case-insensitive filesystem (Windows/NTFS, macOS default), so the
  second write clobbered the first and 5 rows vanished before Bronze ever saw
  them. Bronze is meant to be an append-only, faithful copy of raw data --
  silently losing rows to a filename collision defeats that. Deduplication is
  Silver's job (see silver.py), not something ingest.py should do by accident
  via overwrites.

Usage:
    # Dry run (no AWS account needed yet)
    python ingest.py --input ../data/telemetry_raw.csv --dry-run

    # Real upload (requires AWS credentials configured in your shell/CLI
    # and TELEMETRY_S3_BUCKET set as an environment variable)
    export TELEMETRY_S3_BUCKET=your-bucket-name
    python ingest.py --input ../data/telemetry_raw.csv
"""

import argparse
import io
import os
import sys
import uuid
from pathlib import Path

import pandas as pd

try:
    import boto3
    from botocore.exceptions import NoCredentialsError, ClientError
except ImportError:
    boto3 = None


def build_partitions(df: pd.DataFrame):
    """
    Split the dataframe into (server_id, date, dataframe) partitions.
    Mirrors Hive-style partitioning: server_id=X/dt=YYYY-MM-DD/
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["dt"] = df["timestamp"].dt.strftime("%Y-%m-%d")

    for (server_id, dt), group in df.groupby(["server_id", "dt"]):
        # Drop both -- they're already encoded in the partition folder name that the
        # caller writes this group to (see module docstring).
        partition_df = group.drop(columns=["dt", "server_id"]).reset_index(drop=True)
        yield server_id, dt, partition_df


def partition_filename() -> str:
    """A unique filename per partition write -- see module docstring for why a fixed
    filename is unsafe (it can silently overwrite an earlier write to what a
    case-insensitive filesystem treats as the same folder)."""
    return f"telemetry_{uuid.uuid4().hex[:12]}.parquet"


def to_parquet_bytes(partition_df: pd.DataFrame) -> bytes:
    """Serialize a dataframe partition to Parquet, in-memory (binary)."""
    buffer = io.BytesIO()
    partition_df.to_parquet(buffer, engine="pyarrow", index=False)
    return buffer.getvalue()


def upload_dry_run(server_id: str, dt: str, body: bytes, row_count: int, local_root: str):
    key_path = Path(local_root) / f"server_id={server_id}" / f"dt={dt}" / partition_filename()
    key_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.write_bytes(body)
    print(f"[dry-run] wrote {key_path} ({row_count} records, {len(body):,} bytes)")


def upload_s3(server_id: str, dt: str, body: bytes, row_count: int, bucket: str, prefix: str):
    if boto3 is None:
        print("ERROR: boto3 is not installed. Run: pip install boto3", file=sys.stderr)
        sys.exit(1)

    s3 = boto3.client("s3")  # credentials picked up from env / AWS CLI config
    key = f"{prefix}/server_id={server_id}/dt={dt}/{partition_filename()}"
    try:
        s3.put_object(Bucket=bucket, Key=key, Body=body)
        print(f"[s3] uploaded s3://{bucket}/{key} ({row_count} records, {len(body):,} bytes)")
    except NoCredentialsError:
        print("ERROR: No AWS credentials found. Configure via `aws configure` "
              "or export AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY.", file=sys.stderr)
        sys.exit(1)
    except ClientError as e:
        print(f"ERROR uploading {key}: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Ingest telemetry CSV into partitioned S3 storage.")
    parser.add_argument("--input", default="../data/telemetry_raw.csv", help="Path to generated telemetry CSV")
    parser.add_argument("--dry-run", action="store_true", help="Write to local folder instead of S3")
    parser.add_argument("--local-root", default="../data/s3-mirror/raw", help="Local output root for dry-run mode")
    parser.add_argument("--prefix", default="raw", help="S3 key prefix (folder) for uploaded objects")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: input file not found: {args.input}. Run generate_telemetry.py first.", file=sys.stderr)
        sys.exit(1)

    bucket = os.environ.get("TELEMETRY_S3_BUCKET")
    if not args.dry_run and not bucket:
        print("ERROR: TELEMETRY_S3_BUCKET environment variable is not set.\n"
              "  export TELEMETRY_S3_BUCKET=your-bucket-name   (Git Bash)\n"
              "  $env:TELEMETRY_S3_BUCKET='your-bucket-name'   (PowerShell)\n"
              "Or pass --dry-run to test locally without AWS.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(args.input)
    partition_count = 0

    for server_id, dt, partition_df in build_partitions(df):
        body = to_parquet_bytes(partition_df)
        row_count = len(partition_df)
        if args.dry_run:
            upload_dry_run(server_id, dt, body, row_count, args.local_root)
        else:
            upload_s3(server_id, dt, body, row_count, bucket, args.prefix)
        partition_count += 1

    print(f"\nDone. Ingested {partition_count} partitions "
          f"({len(df):,} total telemetry records).")


if __name__ == "__main__":
    main()
