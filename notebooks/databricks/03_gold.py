# Databricks notebook source
# MAGIC %md
# MAGIC # Gold (Databricks)
# MAGIC
# MAGIC Imports and calls `build_server_summary()` / `build_daily_metrics()` from
# MAGIC `notebooks/gold.py` -- same health-score formula, same disk-trend calculation,
# MAGIC same known limitations (see that module's docstring and
# MAGIC `notebooks/telemetry_analysis.ipynb`'s findings) as the local run.
# MAGIC
# MAGIC Run `01_bronze.py` and `02_silver.py` first.

# COMMAND ----------

dbutils.widgets.text("repo_path", "/Workspace/Repos/you/cloud-telemetry-analytics-yanghan", "Cloned repo path")
dbutils.widgets.text("silver_table", "telemetry_silver", "Silver managed table name")
dbutils.widgets.text("gold_summary_table", "telemetry_gold_server_summary", "Gold summary table name")
dbutils.widgets.text("gold_daily_table", "telemetry_gold_daily_metrics", "Gold daily table name")

REPO_PATH = dbutils.widgets.get("repo_path")
SILVER_TABLE = dbutils.widgets.get("silver_table")
GOLD_SUMMARY_TABLE = dbutils.widgets.get("gold_summary_table")
GOLD_DAILY_TABLE = dbutils.widgets.get("gold_daily_table")

# COMMAND ----------

import sys

sys.path.append(f"{REPO_PATH}/notebooks")
from gold import build_server_summary, build_daily_metrics  # noqa: E402

from pyspark.sql import functions as F

# COMMAND ----------

print(f"Reading Silver from table {SILVER_TABLE} ...")
df = spark.table(SILVER_TABLE)
print(f"Silver row count: {df.count():,}")

# COMMAND ----------

print("\nBuilding gold_server_summary ...")
summary_df = build_server_summary(df)
display(summary_df.orderBy(F.desc("anomaly_count")))

print(f"Writing to table {GOLD_SUMMARY_TABLE} ...")
summary_df.write.format("delta").mode("overwrite").saveAsTable(GOLD_SUMMARY_TABLE)

# COMMAND ----------

print("\nBuilding gold_daily_metrics ...")
daily_df = build_daily_metrics(df)
display(daily_df.limit(10))

print(f"Writing to table {GOLD_DAILY_TABLE} ...")
(
    daily_df.write
    .format("delta")
    .mode("overwrite")
    .partitionBy("server_id")
    .saveAsTable(GOLD_DAILY_TABLE)
)

print("\nGold layer complete.")
