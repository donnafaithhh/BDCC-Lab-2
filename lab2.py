from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lag, abs as ps_abs
from pyspark.sql.window import Window
import matplotlib.pyplot as plt

# --- Create Spark session ---
spark = (
    SparkSession.builder
    .master('local[*]')
    .getOrCreate()
)

# --- Load your Parquet data ---
directory = '/mnt/data/public/binance-full-history'
binance = spark.read.parquet(directory)

# --- Define window for lag calculation ---
w = Window.orderBy("open_time")

# --- Compute percentage return from close prices ---
binance_returns = (
    binance
    .withColumn("prev_close", lag("close").over(w))
    .withColumn("return", (col("close") - col("prev_close")) / col("prev_close"))
    .withColumn("abs_return", ps_abs(col("return")))
    .dropna()
)

# --- Convert small sample to Pandas for visualization ---
sample_df = binance_returns.select("abs_return", "volume").sample(fraction=0.01, seed=42).toPandas()

# --- Create scatter plot ---
plt.figure(figsize=(15,5))
plt.scatter(sample_df["abs_return"], sample_df["volume"], alpha=0.5)
plt.xlabel("Absolute Price Change (|return|)")
plt.ylabel("Trading Volume")
plt.title("Relationship Between Price Movement and Trading Volume")
plt.xscale("log")
plt.yscale("log")
plt.grid(True, which="both", linestyle="--", alpha=0.5)

# --- Save the plot as a PNG file ---
plt.savefig("price_volume_relationship.png", dpi=300, bbox_inches="tight")

# --- Display the plot ---
plt.show()
