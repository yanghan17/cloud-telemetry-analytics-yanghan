"""
gold.py

Builds two Gold Delta tables from Silver:
  - gold_server_summary : one row per server, overall operational metrics
  - gold_daily_metrics  : one row per server per day, for trend charts

Design notes (see chat writeup for full reasoning):
- Reads from Silver ONLY -- aggregating unvalidated data would silently
  corrupt every metric derived from it.
- anomaly_count uses the rule-based `status` field (WARNING/CRITICAL),
  NOT the `is_injected_anomaly` ground-truth column. That ground truth
  only exists because this is synthetic data; a real system wouldn't
  have it, and per the target architecture, Gold is built independently
  of the ML model (ML anomalies flow into Postgres separately, later).
- health_score is a simple, explainable weighted-penalty formula, not a
  statistical/ML score -- that rigor is reserved for the ML step (Day 8).
- disk trend is a cheap first-half-vs-second-half comparison, not a
  fitted regression -- proper trend analysis belongs in the data science
  notebook, not the lakehouse aggregation layer.

Usage:
    python gold.py
"""

from delta import configure_spark_with_delta_pip
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

SILVER_PATH = "../data/lakehouse/silver"
GOLD_SUMMARY_PATH = "../data/lakehouse/gold_server_summary"
GOLD_DAILY_PATH = "../data/lakehouse/gold_daily_metrics"

# Health score formula thresholds/weights -- illustrative operational
# defaults, not derived from a formal SRE framework. Documented here so
# they're easy to explain/tune in review.
CPU_WARN_THRESHOLD = 80
MEMORY_WARN_THRESHOLD = 80
DISK_WARN_THRESHOLD = 90
DISK_PENALTY_WEIGHT = 1.5
ERROR_PENALTY_WEIGHT = 5
ANOMALY_PENALTY_WEIGHT = 2


def build_spark_session() -> SparkSession:
    builder = (
        SparkSession.builder.appName("telemetry-gold")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.driver.extraJavaOptions", "-Duser.timezone=UTC")
        .config("spark.executor.extraJavaOptions", "-Duser.timezone=UTC")
    )
    return configure_spark_with_delta_pip(builder).getOrCreate()


def build_server_summary(df):
    """One row per server: overall aggregated operational metrics."""

    # anomaly_count: rule-based signal only (see module docstring).
    is_anomalous = F.col("status").isin("WARNING", "CRITICAL")

    base = df.groupBy("server_id", "hostname").agg(
        F.avg("cpu_usage_percent").alias("avg_cpu_percent"),
        F.max("cpu_usage_percent").alias("peak_cpu_percent"),
        F.avg("memory_usage_percent").alias("avg_memory_percent"),
        F.max("memory_usage_percent").alias("peak_memory_percent"),
        F.avg("disk_usage_percent").alias("avg_disk_percent"),
        F.max("disk_usage_percent").alias("peak_disk_percent"),
        F.avg("network_rx_bytes").alias("avg_network_rx_bytes"),
        F.avg("network_tx_bytes").alias("avg_network_tx_bytes"),
        F.avg("application_error_count").alias("avg_errors_per_interval"),
        F.avg("response_time_ms").alias("avg_response_time_ms"),
        F.max("response_time_ms").alias("peak_response_time_ms"),
        F.sum(is_anomalous.cast("int")).alias("anomaly_count"),
        F.count("*").alias("total_readings"),
    )

    # errors_per_hour: more human-interpretable than "errors per 5-min interval"
    base = base.withColumn(
        "errors_per_hour", F.round(F.col("avg_errors_per_interval") * 12, 2)
    )

    # --- Disk trend: first-half-of-week avg vs second-half-of-week avg ---
    min_ts = df.agg(F.min("timestamp")).first()[0]
    max_ts = df.agg(F.max("timestamp")).first()[0]
    midpoint = min_ts + (max_ts - min_ts) / 2

    first_half_disk = (
        df.filter(F.col("timestamp") < F.lit(midpoint))
        .groupBy("server_id")
        .agg(F.avg("disk_usage_percent").alias("disk_first_half_avg"))
    )
    second_half_disk = (
        df.filter(F.col("timestamp") >= F.lit(midpoint))
        .groupBy("server_id")
        .agg(F.avg("disk_usage_percent").alias("disk_second_half_avg"))
    )

    base = (
        base
        .join(first_half_disk, "server_id", "left")
        .join(second_half_disk, "server_id", "left")
        .withColumn(
            "disk_trend_percent_change",
            F.round(F.col("disk_second_half_avg") - F.col("disk_first_half_avg"), 2),
        )
        .drop("disk_first_half_avg", "disk_second_half_avg")
    )

    # --- Health score: weighted penalty formula ---
    cpu_penalty = F.greatest(F.lit(0.0), F.col("peak_cpu_percent") - CPU_WARN_THRESHOLD)
    memory_penalty = F.greatest(F.lit(0.0), F.col("peak_memory_percent") - MEMORY_WARN_THRESHOLD)
    disk_penalty = F.greatest(F.lit(0.0), F.col("peak_disk_percent") - DISK_WARN_THRESHOLD) * DISK_PENALTY_WEIGHT
    error_penalty = F.col("avg_errors_per_interval") * ERROR_PENALTY_WEIGHT
    anomaly_penalty = F.col("anomaly_count") * ANOMALY_PENALTY_WEIGHT

    total_penalty = cpu_penalty + memory_penalty + disk_penalty + error_penalty + anomaly_penalty
    health_score = F.greatest(F.lit(0.0), F.least(F.lit(100.0), F.lit(100.0) - total_penalty))

    base = base.withColumn("health_score", F.round(health_score, 1))

    # Round the float metrics for readability
    round_cols = [
        "avg_cpu_percent", "peak_cpu_percent", "avg_memory_percent", "peak_memory_percent",
        "avg_disk_percent", "peak_disk_percent", "avg_network_rx_bytes", "avg_network_tx_bytes",
        "avg_errors_per_interval", "avg_response_time_ms", "peak_response_time_ms",
    ]
    for c in round_cols:
        base = base.withColumn(c, F.round(F.col(c), 2))

    return base


def build_daily_metrics(df):
    """One row per server per day: same core metrics, daily grain, for trend charts."""
    is_anomalous = F.col("status").isin("WARNING", "CRITICAL")

    daily = df.groupBy("server_id", "dt").agg(
        F.avg("cpu_usage_percent").alias("avg_cpu_percent"),
        F.max("cpu_usage_percent").alias("peak_cpu_percent"),
        F.avg("memory_usage_percent").alias("avg_memory_percent"),
        F.max("memory_usage_percent").alias("peak_memory_percent"),
        F.avg("disk_usage_percent").alias("avg_disk_percent"),
        F.max("disk_usage_percent").alias("peak_disk_percent"),
        F.avg("network_rx_bytes").alias("avg_network_rx_bytes"),
        F.avg("network_tx_bytes").alias("avg_network_tx_bytes"),
        F.avg("application_error_count").alias("avg_errors_per_interval"),
        F.avg("response_time_ms").alias("avg_response_time_ms"),
        F.sum(is_anomalous.cast("int")).alias("anomaly_count"),
    )

    round_cols = [c for c in daily.columns if c not in ("server_id", "dt", "anomaly_count")]
    for c in round_cols:
        daily = daily.withColumn(c, F.round(F.col(c), 2))

    return daily.orderBy("server_id", "dt")


def main():
    spark = build_spark_session()
    spark.sparkContext.setLogLevel("WARN")

    print(f"Reading Silver from {SILVER_PATH} ...")
    df = spark.read.format("delta").load(SILVER_PATH)
    print(f"Silver row count: {df.count():,}")

    print("\nBuilding gold_server_summary ...")
    summary_df = build_server_summary(df)
    summary_df.orderBy(F.desc("anomaly_count")).show(20, truncate=False)

    print(f"Writing to {GOLD_SUMMARY_PATH} ...")
    summary_df.write.format("delta").mode("overwrite").save(GOLD_SUMMARY_PATH)

    print("\nBuilding gold_daily_metrics ...")
    daily_df = build_daily_metrics(df)
    daily_df.show(10, truncate=False)

    print(f"Writing to {GOLD_DAILY_PATH} ...")
    (
        daily_df.write
        .format("delta")
        .mode("overwrite")
        .partitionBy("server_id")
        .save(GOLD_DAILY_PATH)
    )

    print("\nGold layer complete.")
    spark.stop()


if __name__ == "__main__":
    main()
