# Databricks notebook source
# MAGIC %md
# MAGIC # Bronze (Databricks)
# MAGIC
# MAGIC Databricks-native runner for the Bronze layer. It does not reimplement the
# MAGIC transformation -- it imports `build_bronze()` from `notebooks/bronze.py` in this
# MAGIC same repo (cloned into the workspace via Git folders) and calls it, so the local
# MAGIC and Databricks paths are guaranteed to do the exact same thing.
# MAGIC
# MAGIC ## Why this exists as a separate notebook rather than running bronze.py directly
# MAGIC
# MAGIC `bronze.py`'s `main()` builds its own local `SparkSession` (with Windows-specific
# MAGIC networking config) and reads from a local filesystem mirror of S3. On Databricks:
# MAGIC - `spark` already exists as a global -- building a new session is not just
# MAGIC   unnecessary, it's unsupported on serverless compute.
# MAGIC - Free Edition's serverless compute has no CLI/API/instance-profile access to S3
# MAGIC   (see [Databricks Free Edition limitations](https://docs.databricks.com/aws/en/getting-started/free-edition-limitations)),
# MAGIC   so this reads from a **Unity Catalog Volume** instead -- upload the same raw
# MAGIC   Parquet partitions there once (see the README section this notebook was written
# MAGIC   for). A paid workspace with an instance profile would instead point
# MAGIC   `RAW_VOLUME_PATH` at `s3a://your-bucket/raw/server_id=*/dt=*/` directly, and
# MAGIC   nothing else in this notebook would change.
# MAGIC
# MAGIC ## Prerequisites (one-time, manual, done in the Databricks UI)
# MAGIC 1. Clone this GitHub repo into the workspace: **Workspace -> Git folders -> Add** ->
# MAGIC    paste the repo URL.
# MAGIC 2. Create a Unity Catalog Volume and upload the raw partitions into it, preserving
# MAGIC    the `server_id=X/dt=Y/*.parquet` folder structure -- e.g. drag-and-drop the
# MAGIC    whole `data/s3-mirror/raw/` folder from `ingest.py --dry-run`'s output.
# MAGIC 3. Set the widgets below to match your paths, then **Run All**.

# COMMAND ----------

dbutils.widgets.text("repo_path", "/Workspace/Users/yteh0009@zohomail.com/cloud-telemetry-analytics-yanghan", "Cloned repo path")
dbutils.widgets.text("raw_volume_path", "/Volumes/workspace/default/telemetry/raw", "Raw input volume path")
dbutils.widgets.text("bronze_table", "telemetry_bronze", "Bronze managed table name")

# Widgets survive a Python restart; plain variables don't -- restarting here (once,
# before anything imports notebooks/bronze.py) guarantees sys.path.append below is
# picked up on serverless compute, without needing a second manual "Run All".
dbutils.library.restartPython()

# COMMAND ----------

import sys

REPO_PATH = dbutils.widgets.get("repo_path")
RAW_VOLUME_PATH = dbutils.widgets.get("raw_volume_path")
BRONZE_TABLE = dbutils.widgets.get("bronze_table")

sys.path.append(f"{REPO_PATH}/notebooks")
from bronze import build_bronze  # noqa: E402


# COMMAND ----------

print(f"Reading raw partitions from {RAW_VOLUME_PATH} ...")
spark.conf.set("spark.sql.session.timeZone", "UTC")
df = (
    spark.read
    .option("basePath", RAW_VOLUME_PATH)
    .parquet(f"{RAW_VOLUME_PATH}/server_id=*/dt=*/")
)

row_count = df.count()
print(f"Read {row_count:,} raw rows across {df.select('server_id').distinct().count()} servers.")

# COMMAND ----------

bronze_df = build_bronze(df)

print(f"Writing Bronze managed table {BRONZE_TABLE} ...")
(
    bronze_df.write
    .format("delta")
    .mode("overwrite")
    .partitionBy("server_id")
    .saveAsTable(BRONZE_TABLE)
)

print("Bronze layer complete.")
display(spark.table(BRONZE_TABLE).limit(20))
