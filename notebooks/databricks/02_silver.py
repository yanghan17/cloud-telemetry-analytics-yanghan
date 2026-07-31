# Databricks notebook source
# MAGIC %md
# MAGIC # Silver (Databricks)
# MAGIC
# MAGIC Imports and calls `clean_and_validate()` from `notebooks/silver.py` -- the exact
# MAGIC same function `tests/test_silver.py` unit-tests locally, including the
# MAGIC standardize-before-deduplicate fix and all six rejection rules. Nothing about the
# MAGIC validation logic changes on Databricks; only where Bronze/Silver are read from and
# MAGIC written to does.
# MAGIC
# MAGIC Run `01_bronze.py` first.

# COMMAND ----------

dbutils.widgets.text("repo_path", "/Workspace/Users/yteh0009@zohomail.com/cloud-telemetry-analytics-yanghan", "Cloned repo path")
dbutils.widgets.text("bronze_table", "telemetry_bronze", "Bronze managed table name")
dbutils.widgets.text("silver_table", "telemetry_silver", "Silver managed table name")
dbutils.widgets.text("rejects_table", "telemetry_rejects", "Rejects managed table name")

# Widgets survive a Python restart; plain variables don't -- restarting here (once,
# before anything imports notebooks/silver.py) guarantees sys.path.append below is
# picked up on serverless compute, without needing a second manual "Run All".
dbutils.library.restartPython()

# COMMAND ----------

import sys

REPO_PATH = dbutils.widgets.get("repo_path")
BRONZE_TABLE = dbutils.widgets.get("bronze_table")
SILVER_TABLE = dbutils.widgets.get("silver_table")
REJECTS_TABLE = dbutils.widgets.get("rejects_table")

sys.path.append(f"{REPO_PATH}/notebooks")
from silver import clean_and_validate  # noqa: E402

from pyspark.sql import functions as F

# COMMAND ----------

print(f"Reading Bronze from table {BRONZE_TABLE} ...")
spark.conf.set("spark.sql.session.timeZone", "UTC")
df = spark.table(BRONZE_TABLE)
bronze_count = df.count()
print(f"Bronze row count: {bronze_count:,}")

valid_df, rejected_df = clean_and_validate(df)

valid_count = valid_df.count()
rejected_count = rejected_df.count()

print("\nValidation results:")
print(f"  Valid rows:    {valid_count:,}")
print(f"  Rejected rows: {rejected_count:,}")
if rejected_count > 0:
    print("\n  Rejection breakdown:")
    rejected_df.groupBy("_reject_reason").count().orderBy(F.desc("count")).show(truncate=False)

# COMMAND ----------

print(f"Writing Silver managed table {SILVER_TABLE} ...")
(
    valid_df.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .partitionBy("server_id")
    .saveAsTable(SILVER_TABLE)
)

if rejected_count > 0:
    print(f"Writing rejects table {REJECTS_TABLE} ...")
    (
        rejected_df.write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(REJECTS_TABLE)
    )

print(f"\nSilver layer complete. {valid_count:,}/{bronze_count:,} rows passed validation "
      f"({100 * valid_count / bronze_count:.1f}%).")
display(spark.table(SILVER_TABLE).limit(20))
