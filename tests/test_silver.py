"""
test_silver.py

Unit tests for silver.py's `clean_and_validate` transform, run against small,
hand-built DataFrames rather than a full Bronze/S3 pipeline run -- fast, and able to
exercise edge cases (like the server_id case-variant below) that can't reliably be
reproduced end-to-end through a CSV -> S3 -> Bronze path on a case-insensitive local
filesystem (see generate_telemetry.py's `inject_data_quality_issues` docstring).

Requires a local Spark session, same as the pipeline scripts -- these are not fast
unit tests in the "milliseconds" sense (Spark session startup dominates), but they
exercise the real Silver logic instead of a reimplementation of it, which is the
more important property for a validation layer like this one.

Usage:
    pytest tests/test_silver.py -v
"""

import os
import sys
from datetime import datetime, timezone

import pytest
from pyspark.sql import Row, SparkSession

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "notebooks"))
from silver import clean_and_validate, MIN_VALID_TIMESTAMP  # noqa: E402


@pytest.fixture(scope="module")
def spark():
    session = (
        SparkSession.builder.appName("telemetry-silver-tests")
        .master("local[1]")
        .config("spark.sql.shuffle.partitions", "1")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.driver.extraJavaOptions", "-Duser.timezone=UTC")
        .config("spark.executor.extraJavaOptions", "-Duser.timezone=UTC")
        .getOrCreate()
    )
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()


def make_row(**overrides):
    """A minimally valid Bronze-shaped row, with overrides for the field(s) under test."""
    base = dict(
        timestamp=datetime(2026, 7, 15, 12, 0, 0, tzinfo=timezone.utc),
        server_id="server01",
        hostname="web-app-01",
        cpu_usage_percent=40.0,
        memory_usage_percent=50.0,
        disk_usage_percent=60.0,
        disk_io_read=100.0,
        disk_io_write=80.0,
        network_rx_bytes=500_000.0,
        network_tx_bytes=300_000.0,
        application_error_count=1,
        response_time_ms=120.0,
        status="OK",
    )
    base.update(overrides)
    return Row(**base)


def test_valid_row_passes_through(spark):
    df = spark.createDataFrame([make_row()])
    valid_df, rejected_df = clean_and_validate(df)
    assert valid_df.count() == 1
    assert rejected_df.count() == 0


def test_cpu_over_100_is_rejected(spark):
    df = spark.createDataFrame([make_row(cpu_usage_percent=140.0)])
    valid_df, rejected_df = clean_and_validate(df)
    assert valid_df.count() == 0
    assert rejected_df.count() == 1
    assert rejected_df.first()["_reject_reason"] == "cpu_out_of_range"


def test_negative_network_bytes_is_rejected(spark):
    df = spark.createDataFrame([make_row(network_rx_bytes=-500.0)])
    valid_df, rejected_df = clean_and_validate(df)
    assert valid_df.count() == 0
    assert rejected_df.first()["_reject_reason"] == "negative_io_or_network"


def test_out_of_bounds_timestamp_is_rejected(spark):
    # A future date, not a pre-1970 one: Windows' C runtime can't represent dates
    # before the epoch in the path PySpark's schema inference takes, which is a
    # platform limitation, not something silver.py needs to handle.
    far_future = datetime(2099, 1, 1, tzinfo=timezone.utc)
    df = spark.createDataFrame([make_row(timestamp=far_future)])
    valid_df, rejected_df = clean_and_validate(df)
    assert valid_df.count() == 0
    assert rejected_df.first()["_reject_reason"] == "invalid_timestamp"
    assert far_future > MIN_VALID_TIMESTAMP  # sanity-check: rejected for being too late, not too early


def test_exact_duplicate_collapses_to_one_row(spark):
    row = make_row()
    df = spark.createDataFrame([row, row])
    valid_df, rejected_df = clean_and_validate(df)
    assert valid_df.count() == 1
    assert rejected_df.count() == 0


def test_server_id_case_variant_duplicate_collapses_to_one_row(spark):
    """
    The regression test for the real bug this project found: two readings for the
    same (server, instant) that differ only in server_id casing/whitespace must
    collapse to exactly one valid row, not two.

    Before the fix (deduplicate, THEN standardize): "SERVER01" and "server01" look
    like different servers to dropDuplicates, so both survive as "valid" -- Gold and
    Postgres would silently double-count this one reading.

    After the fix (standardize, THEN deduplicate): both rows normalize to server_id
    "server01" before dropDuplicates ever runs, so they correctly collapse to one.
    """
    row_a = make_row(server_id="server01")
    row_b = make_row(server_id="  SERVER01  ")  # same instant, same server, different raw casing
    df = spark.createDataFrame([row_a, row_b])

    valid_df, rejected_df = clean_and_validate(df)

    assert valid_df.count() == 1, (
        "Expected the case-variant duplicate to collapse to a single row -- "
        "if this fails, standardization is running after deduplication again."
    )
    assert rejected_df.count() == 0
    assert valid_df.first()["server_id"] == "server01"


def test_missing_required_metric_is_rejected(spark):
    # A second, fully-valid row alongside the null one -- Spark's schema inference
    # can't determine memory_usage_percent's type from a single all-null column.
    df = spark.createDataFrame([make_row(), make_row(server_id="server02", memory_usage_percent=None)])
    valid_df, rejected_df = clean_and_validate(df)
    assert valid_df.count() == 1
    assert rejected_df.count() == 1
    assert rejected_df.first()["_reject_reason"] == "memory_out_of_range"


def test_mixed_batch_only_rejects_the_bad_rows(spark):
    """A batch of mostly-valid data with exactly one defect should reject exactly one row."""
    rows = [make_row(server_id=f"server{i:02d}") for i in range(1, 6)]
    rows.append(make_row(server_id="server06", application_error_count=-5))
    df = spark.createDataFrame(rows)

    valid_df, rejected_df = clean_and_validate(df)

    assert valid_df.count() == 5
    assert rejected_df.count() == 1
    assert rejected_df.first()["_reject_reason"] == "negative_errors_or_latency"
