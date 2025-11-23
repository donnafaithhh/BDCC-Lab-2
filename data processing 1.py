# standard imports
import matplotlib.pyplot as plt
import random
import pandas as pd
import glob
import os
import numpy as np
from functools import reduce

# spark imports
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import FloatType
import pyspark.sql.window as window
from pyspark.sql.functions import col, split, when, lit
from pyspark.sql.functions import regexp_extract, input_file_name
from pyspark.sql.functions import year, month, dayofmonth, to_date

# Directory containing the parquet files
directory = '/mnt/data/public/binance-full-history'
files = [f for f in os.listdir(directory) if f.endswith('.parquet')]

base_quote_counts = {}

for file in files:
    # Get base and quote per file name
    base_currency = file.split('-')[0]
    quote_currency = file.split('-')[1].replace('.parquet', '')
    
    if base_currency not in base_quote_counts:
        base_quote_counts[base_currency] = set()
    
    base_quote_counts[base_currency].add(quote_currency)

# Convert to counts
base_counts = {base: len(quotes) for base, quotes in base_quote_counts.items()}

# Create dataframe for plotting
counts_df = pd.DataFrame({
    'base_currency': list(base_counts.keys()),
    'quote_count': list(base_counts.values())
})
quote_count_distribution = counts_df['quote_count'].value_counts().sort_index()

# Get top base currencies
top_bases = counts_df.nlargest(10, 'quote_count')
common_quotes = set.intersection(*(base_quote_counts[base] for base in top_bases['base_currency']))

# Initialize Spark Session
spark = SparkSession.builder.appName("TopBasesAnalysis").getOrCreate()
directory = '/mnt/data/public/binance-full-history'

# Getting only target files
file_data = []
for base in top_bases['base_currency']:
    for quote in common_quotes:
        file_name = f"{base}-{quote}.parquet"
        file_path = os.path.join(directory, file_name)
        if os.path.exists(file_path):
            file_data.append((file_path, base, quote))

# Define classification mapping
classification_map = {
    'stablecoin': ['BUSD', 'USDT'],
    'fiat': ['EUR']
}

# Create DataFrames for each file and add columns
dfs = []
for file_path, base, quote in file_data:
    df = spark.read.parquet(file_path)
    df = df.withColumn("base_currency", lit(base)) \
           .withColumn("quote_currency", lit(quote)) \
           .withColumn("classification", 
                      when(lit(quote).isin(classification_map['stablecoin']), lit('stablecoin'))
                      .when(lit(quote).isin(classification_map['fiat']), lit('fiat'))
                      .otherwise(lit('cryptocurrency')))
    dfs.append(df)

# Union all DataFrames
spark_df = reduce(lambda df1, df2: df1.union(df2), dfs)

print(f"Loaded {len(file_data)} parquet files.")








import os
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
import pyspark.sql.window as window
from pyspark.sql import functions as F
from pyspark.sql.functions import col, to_date, row_number, lit

def calculate_volatility_measures(
    spark_df,
    output_dir='Figures',
    rolling_window_days=30,
    annualization_factor=np.sqrt(365),
    min_rolling_obs=10
):
    """
    Produces:
      - annualized volatility per year (using daily log returns)
      - rolling (N-day) annualized volatility time series
    Saves:
      - 'annual_volatility_by_year.png'
      - 'rolling_volatility_examples.png'
    Returns:
      - annual_vol_pd : pandas DataFrame with columns [base_currency, quote_currency, year, classification, annual_volatility]
      - rolling_vol_pd: pandas DataFrame with rolling vol time series (date-level)
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1) Ensure there's a date column (UTC day). Use last close of the day.
    spark_daily = (
        spark_df
        .withColumn('date', to_date('open_time'))  # extract date from timestamp
    )

    # pick last close per (base,quote,date) by ordering open_time desc
    rn_window = window.Window.partitionBy('base_currency', 'quote_currency', 'date').orderBy(F.col('open_time').desc())
    spark_daily = (
        spark_daily
        .withColumn('_rn', row_number().over(rn_window))
        .filter(col('_rn') == 1)
        .drop('_rn')
    )

    # 2) Compute daily log returns (lag on date)
    lag_window = window.Window.partitionBy('base_currency', 'quote_currency').orderBy('date')
    spark_daily = (
        spark_daily
        .withColumn('prev_close', F.lag('close').over(lag_window))
        .withColumn('log_return', F.when(F.col('prev_close').isNotNull(),
                                         F.log(col('close') / col('prev_close')))
                     .otherwise(F.lit(None)))
    )

    # Filter out null/inf returns
    spark_daily = spark_daily.filter(col('log_return').isNotNull())

    # 3) Add year column
    spark_daily = spark_daily.withColumn('year', F.year(F.col('date')))

    # 4) Annualized volatility per year: stddev(daily_log_return) * sqrt(365)
    annual_vol_df = (
        spark_daily
        .groupBy('base_currency', 'quote_currency', 'year', 'classification')
        .agg(
            (F.stddev('log_return') * F.lit(annualization_factor)).alias('annual_volatility'),
            F.count('log_return').alias('n_days')  # helpful to filter small-sample years
        )
        .filter(col('annual_volatility').isNotNull())
        .orderBy('base_currency', 'quote_currency', 'year')
    )

    # 5) Rolling volatility (N-day) at daily frequency (use rowsBetween with a window over ordered dates)
    # We'll compute rolling stddev over the last `rolling_window_days` days (including current), then annualize.
    rolling_w = window.Window.partitionBy('base_currency', 'quote_currency').orderBy('date').rowsBetween(-(rolling_window_days-1), 0)
    spark_daily = spark_daily.withColumn('rolling_std', F.stddev('log_return').over(rolling_w))
    spark_daily = spark_daily.withColumn('rolling_vol_annualized',
                                         F.when(F.col('rolling_std').isNotNull(), F.col('rolling_std') * F.lit(annualization_factor))
                                         .otherwise(None))
    # Count non-null returns in rolling window to ensure we have enough observations
    count_w = window.Window.partitionBy('base_currency', 'quote_currency').orderBy('date').rowsBetween(-(rolling_window_days-1), 0)
    spark_daily = spark_daily.withColumn('rolling_count', F.count('log_return').over(count_w))
    spark_daily = spark_daily.withColumn('rolling_vol_annualized',
                                         F.when(col('rolling_count') >= min_rolling_obs, col('rolling_vol_annualized'))
                                         .otherwise(F.lit(None)))

    # 6) Convert results to pandas for plotting (careful with memory; you may sample or filter to top markets)
    annual_vol_pd = annual_vol_df.toPandas()
    rolling_pd = spark_daily.select('base_currency', 'quote_currency', 'date', 'rolling_vol_annualized', 'rolling_count', 'classification').toPandas()

    # 7) Basic plotting
    # 7a: Annual volatility by year (one plot per base_currency, up to 10 bases)
    try:
        import math
        base_list = sorted(annual_vol_pd['base_currency'].unique())
        n_plots = min(len(base_list), 10)
        cols = 5
        rows = math.ceil(n_plots / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(cols*5, rows*4))
        axes = axes.flatten()
        all_years = sorted(annual_vol_pd['year'].unique())
        if len(all_years) == 0:
            print("No annual data to plot.")
        else:
            for i, base in enumerate(base_list[:n_plots]):
                ax = axes[i]
                bdf = annual_vol_pd[annual_vol_pd['base_currency'] == base]
                for quote in sorted(bdf['quote_currency'].unique()):
                    qdf = bdf[bdf['quote_currency'] == quote]
                    ax.plot(qdf['year'], qdf['annual_volatility'], marker='o', label=quote)
                ax.set_title(f'{base} annualized vol')
                ax.set_xlabel('Year')
                ax.set_ylabel('Annual volatility')
                ax.set_xticks(all_years)
                ax.grid(alpha=0.3)
                ax.legend(fontsize=8)
            # hide unused
            for j in range(n_plots, len(axes)):
                axes[j].set_visible(False)
            plt.suptitle('Annualized Volatility by Base Currency (per quote)', fontsize=14, y=0.98)
            plt.tight_layout(rect=[0,0,1,0.97])
            out1 = os.path.join(output_dir, 'annual_volatility_by_year.png')
            plt.savefig(out1, dpi=300, bbox_inches='tight')
            plt.show()
            print("Saved:", out1)
    except Exception as e:
        print("Plotting annual volatility failed:", e)

    # 7b: Rolling volatility examples — plot a few (top 4) base-quote pairs by data size
    try:
        # pick top 4 series by count
        counts = rolling_pd.groupby(['base_currency', 'quote_currency']).size().reset_index(name='n')
        top_series = counts.sort_values('n', ascending=False).head(4)
        fig, axes = plt.subplots(len(top_series), 1, figsize=(12, 3*len(top_series)))
        if len(top_series) == 1:
            axes = [axes]
        for ax, (_, row) in zip(axes, top_series.iterrows()):
            base = row['base_currency']
            quote = row['quote_currency']
            ser = rolling_pd[(rolling_pd['base_currency'] == base) & (rolling_pd['quote_currency'] == quote)].sort_values('date')
            ax.plot(ser['date'], ser['rolling_vol_annualized'], marker='.', linewidth=1)
            ax.set_title(f'Rolling {rolling_window_days}-day annualized vol: {base}-{quote}')
            ax.set_ylabel('Annualized vol')
            ax.set_xlabel('Date')
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        out2 = os.path.join(output_dir, 'rolling_volatility_examples.png')
        plt.savefig(out2, dpi=300, bbox_inches='tight')
        plt.show()
        print("Saved:", out2)
    except Exception as e:
        print("Plotting rolling volatility failed:", e)

    return annual_vol_pd, rolling_pd

# Usage: (this will run the improved pipeline and return pandas DataFrames)
annual_volatility_df, rolling_volatility_df = calculate_volatility_measures(spark_df)
