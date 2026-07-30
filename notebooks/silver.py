"""
silver.py

Builds the Silver Delta table: cleaned, validated, deduplicated telemetry.
Invalid rows are NOT silently dropped -- they're written to a separate
`telemetry_rejects` Delta table for auditability (see chat writeup).

Silver responsibilities (per assessment spec):
- Remove duplicate records
- Handle missing values
- Correct data types
- Standardize server names (hostname)
- Validate timestamps
- Reject CPU values above 100% (and other impossible percentages)
- Reject negative telemetry values

Step order: cast types -> standardize strings -> deduplicate -> validate -> split.
------------------------------------------------------------------------------------
This used to run deduplicate -> standardize -- fixed after
telemetry-generator/generate_telemetry.py's `inject_data_quality_issues` added a
"case_variant_duplicate" test case and caught a real bug: a reading with server_id
"SERVER02" and the same reading with server_id "server02" (same timestamp) look like
two *different* servers to `dropDuplicates` before standardization runs, so both
survived deduplication as distinct rows. Only afterward, once `.trim().lower()`
normalized both to "server02", did they become true duplicates -- but by then Silver
had already committed to keeping both, so Gold and Postgres would have silently
double-counted that one reading. Standardizing first means dedup always compares
already-canonical keys, so this class of duplicate can't slip through.

Usage:
    python silver.py
"""

from datetime import datetime, timezone
from delta import configure_spark_with_delta_pip
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, LongType, StringType, TimestampType

BRONZE_PATH = "../data/lakehouse/bronze"
SILVER_PATH = "../data/lakehouse/silver"
REJECTS_PATH = "../data/lakehouse/telemetry_rejects"

# Telemetry generation window sanity bounds -- guards against clock drift /
# corrupted timestamps rather than assuming exact generator dates, so this
# stays valid even if the generator's date range changes later.
MIN_VALID_TIMESTAMP = datetime(2020, 1, 1, tzinfo=timezone.utc)


def build_spark_session() -> SparkSession:
    builder = (
        SparkSession.builder.appName("telemetry-silver")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("spark.sql.shuffle.partitions", "4")
        # Windows sometimes resolves the machine hostname to an IP Spark's
        # internal driver<->executor networking can't bind to, causing
        # BlockManagerId/NullPointerException loops. Forcing localhost
        # fixes this for single-machine local runs.
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
    )
    return configure_spark_with_delta_pip(builder).getOrCreate()


def clean_and_validate(df, min_valid_timestamp: datetime = MIN_VALID_TIMESTAMP):
    """
    The actual Silver transformation, factored out of main() so it can be unit
    tested against a small, hand-built DataFrame (see tests/test_silver.py) without
    needing Bronze, S3, or a full pipeline run. Takes a Bronze-shaped DataFrame,
    returns (valid_df, rejected_df).
    """
    # --- 1. Correct / enforce data types explicitly (before anything compares values) ---
    df = (
        df
        .withColumn("timestamp", F.col("timestamp").cast(TimestampType()))
        .withColumn("cpu_usage_percent", F.col("cpu_usage_percent").cast(DoubleType()))
        .withColumn("memory_usage_percent", F.col("memory_usage_percent").cast(DoubleType()))
        .withColumn("disk_usage_percent", F.col("disk_usage_percent").cast(DoubleType()))
        .withColumn("disk_io_read", F.col("disk_io_read").cast(DoubleType()))
        .withColumn("disk_io_write", F.col("disk_io_write").cast(DoubleType()))
        .withColumn("network_rx_bytes", F.col("network_rx_bytes").cast(DoubleType()))
        .withColumn("network_tx_bytes", F.col("network_tx_bytes").cast(DoubleType()))
        .withColumn("application_error_count", F.col("application_error_count").cast(LongType()))
        .withColumn("response_time_ms", F.col("response_time_ms").cast(DoubleType()))
        .withColumn("status", F.col("status").cast(StringType()))
    )

    # --- 2. Standardize hostname / server_id BEFORE deduplicating (see module docstring) ---
    df = (
        df
        .withColumn("hostname", F.trim(F.lower(F.col("hostname"))))
        .withColumn("server_id", F.trim(F.lower(F.col("server_id"))))
    )

    # --- 3. Deduplicate on the now-canonical business key, not full row ---
    # Keeps the first occurrence per (server_id, timestamp); drops the rest.
    df = df.dropDuplicates(["server_id", "timestamp"])

    # --- 4. Build validity flags (each rule named, so rejects are explainable) ---
    validity_checks = {
        "missing_required_field": (
            F.col("server_id").isNull() | F.col("timestamp").isNull()
        ),
        "invalid_timestamp": (
            F.col("timestamp").isNull()
            | (F.col("timestamp") < F.lit(min_valid_timestamp))
            | (F.col("timestamp") > F.current_timestamp())
        ),
        "cpu_out_of_range": (
            F.col("cpu_usage_percent").isNull()
            | (F.col("cpu_usage_percent") < 0) | (F.col("cpu_usage_percent") > 100)
        ),
        "memory_out_of_range": (
            F.col("memory_usage_percent").isNull()
            | (F.col("memory_usage_percent") < 0) | (F.col("memory_usage_percent") > 100)
        ),
        "disk_out_of_range": (
            F.col("disk_usage_percent").isNull()
            | (F.col("disk_usage_percent") < 0) | (F.col("disk_usage_percent") > 100)
        ),
        "negative_io_or_network": (
            (F.col("disk_io_read") < 0) | (F.col("disk_io_write") < 0)
            | (F.col("network_rx_bytes") < 0) | (F.col("network_tx_bytes") < 0)
        ),
        "negative_errors_or_latency": (
            (F.col("application_error_count") < 0) | (F.col("response_time_ms") < 0)
        ),
    }

    # Combine all checks into one "rejection reason" column: the first rule
    # that fails wins, so every rejected row has a clear, single explanation.
    reject_reason = F.lit(None).cast(StringType())
    for reason, is_bad in validity_checks.items():
        reject_reason = F.when(is_bad, F.lit(reason)).otherwise(reject_reason)

    df = df.withColumn("_reject_reason", reject_reason)

    valid_df = df.filter(F.col("_reject_reason").isNull()).drop("_reject_reason")
    rejected_df = df.filter(F.col("_reject_reason").isNotNull())
    return valid_df, rejected_df


def main():
    spark = build_spark_session()
    spark.sparkContext.setLogLevel("WARN")

    print(f"Reading Bronze from {BRONZE_PATH} ...")
    df = spark.read.format("delta").load(BRONZE_PATH)
    bronze_count = df.count()
    print(f"Bronze row count: {bronze_count:,}")

    valid_df, rejected_df = clean_and_validate(df)

    valid_count = valid_df.count()
    rejected_count = rejected_df.count()

    print(f"\nValidation results:")
    print(f"  Valid rows:    {valid_count:,}")
    print(f"  Rejected rows: {rejected_count:,}")
    if rejected_count > 0:
        print(f"\n  Rejection breakdown:")
        rejected_df.groupBy("_reject_reason").count().orderBy(F.desc("count")).show(truncate=False)

    # --- 5. Write Silver (valid rows) ---
    print(f"Writing Silver Delta table to {SILVER_PATH} ...")
    (
        valid_df.write
        .format("delta")
        .mode("overwrite")
        .partitionBy("server_id")
        .save(SILVER_PATH)
    )

    # --- 6. Write rejects (quarantine, for auditability) ---
    if rejected_count > 0:
        print(f"Writing rejected rows to {REJECTS_PATH} ...")
        (
            rejected_df.write
            .format("delta")
            .mode("overwrite")
            .save(REJECTS_PATH)
        )

    print(f"\nSilver layer complete. "
          f"{valid_count:,}/{bronze_count:,} rows passed validation "
          f"({100 * valid_count / bronze_count:.1f}%).")

    spark.stop()


if __name__ == "__main__":
    main()
