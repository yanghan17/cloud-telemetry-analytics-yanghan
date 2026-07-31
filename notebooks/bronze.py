"""
bronze.py

Builds the Bronze Delta table: raw telemetry, minimally transformed.

Design notes (see chat writeup for full reasoning):
- Reads locally-staged Parquet partitions (synced down from S3 via
  `aws s3 sync`) rather than reading s3a:// directly from Spark, to avoid
  the extra hadoop-aws connector setup on top of an already Windows-heavy
  local Spark environment. Production would point this at s3a://bucket/raw/
  instead -- only the input path changes, not the transformation logic.
- Adds lineage columns (_source_file, _ingested_at) but does NOT clean,
  filter, or validate anything -- that belongs in Silver. Bronze should
  stay a faithful, reprocessable copy of the raw data.
- Partitioned by server_id, since most downstream queries in this project
  are per-server (anomaly investigation, health dashboards).

Usage:
    python bronze.py
"""

import os
from delta import configure_spark_with_delta_pip
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

RAW_INPUT_PATH = "../data/s3-mirror/raw"
BRONZE_OUTPUT_PATH = "../data/lakehouse/bronze"


def build_spark_session() -> SparkSession:
    """
    Configure a local Spark session with Delta Lake support.
    configure_spark_with_delta_pip handles fetching the correct Delta JARs
    to match the installed pyspark version.
    """
    builder = (
        SparkSession.builder.appName("telemetry-bronze")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        # Local dev only needs 1 shuffle partition worth of parallelism at this data size;
        # Spark's default of 200 would create excessive tiny output files for ~20K rows.
        .config("spark.sql.shuffle.partitions", "4")
        # Windows sometimes resolves the machine hostname to an IP Spark's
        # internal driver<->executor networking can't bind to, causing
        # BlockManagerId/NullPointerException loops. Forcing localhost
        # fixes this for single-machine local runs.
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        # Naive / local-session timestamps previously shifted by the OS offset;
        # keep lakehouse instants identical to the UTC-aware generator output.
        .config("spark.sql.session.timeZone", "UTC")
        # Also pin the JVM default TZ so Python `.collect()` / `.first()` datetimes
        # match UTC (otherwise Asia/Kuala_Lumpur makes midnight UTC print as 08:00).
        .config("spark.driver.extraJavaOptions", "-Duser.timezone=UTC")
        .config("spark.executor.extraJavaOptions", "-Duser.timezone=UTC")
    )
    return configure_spark_with_delta_pip(builder).getOrCreate()


def build_bronze(df):
    """
    The actual Bronze transformation, factored out of main() so the Databricks
    notebook version (notebooks/databricks/01_bronze.py) can import and call it
    directly against the same input read a different way (Unity Catalog Volume
    instead of a local mirror), rather than duplicating this logic.

    Uses the `_metadata.file_path` column rather than `input_file_name()` --
    both resolve to the same lineage information, but `input_file_name()` is
    blocked outright on Databricks/Unity Catalog ("UC_COMMAND_NOT_SUPPORTED"),
    while `_metadata` is a standard Spark 3.2+ file-source column that works
    identically locally and on Databricks.
    """
    return (
        df
        .withColumn("_source_file", F.col("_metadata.file_path"))
        .withColumn("_ingested_at", F.current_timestamp())
    )


def main():
    if not os.path.exists(RAW_INPUT_PATH):
        raise FileNotFoundError(
            f"Raw input not found at {RAW_INPUT_PATH}. "
            f"Run: aws s3 sync s3://<your-bucket>/raw {RAW_INPUT_PATH}"
        )

    spark = build_spark_session()
    spark.sparkContext.setLogLevel("WARN")  # Spark's default INFO logging is very noisy

    print(f"Reading raw partitions from {RAW_INPUT_PATH} ...")

    # basePath tells Spark where Hive-style partition discovery should start,
    # so server_id= and dt= folder names become real DataFrame columns
    # instead of just being part of the file path string.
    df = (
        spark.read
        .option("basePath", RAW_INPUT_PATH)
        .parquet(f"{RAW_INPUT_PATH}/server_id=*/dt=*/")
    )

    row_count = df.count()
    print(f"Read {row_count:,} raw rows across {df.select('server_id').distinct().count()} servers.")

    # Lineage / audit columns -- Bronze's job is traceability, not cleaning.
    bronze_df = build_bronze(df)

    print(f"Writing Bronze Delta table to {BRONZE_OUTPUT_PATH} ...")
    (
        bronze_df.write
        .format("delta")
        .mode("overwrite")
        .partitionBy("server_id")
        .save(BRONZE_OUTPUT_PATH)
    )

    print("Bronze layer complete.")
    print("Schema:")
    bronze_df.printSchema()

    spark.stop()


if __name__ == "__main__":
    main()
