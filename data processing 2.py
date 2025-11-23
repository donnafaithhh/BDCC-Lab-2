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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.functions import col, to_date, row_number
from scipy.stats import spearmanr


def analyze_volume_volatility(spark_df, output_dir='Figures'):
    os.makedirs(output_dir, exist_ok=True)

    # Convert timestamps to dates
    spark_daily = spark_df.withColumn('date', to_date('open_time'))

    # Keep the last candle of each day
    rn_window = Window.partitionBy(
        'base_currency', 'quote_currency', 'date'
    ).orderBy(F.col('open_time').desc())

    spark_daily = (
        spark_daily
        .withColumn('_rn', row_number().over(rn_window))
        .filter(col('_rn') == 1)
        .drop('_rn')
    )

    # Compute simple daily returns
    lag_window = Window.partitionBy(
        'base_currency', 'quote_currency'
    ).orderBy('date')

    spark_daily = (
        spark_daily
        .withColumn('prev_close', F.lag('close').over(lag_window))
        .withColumn(
            'return',
            F.when(
                F.col('prev_close').isNotNull(),
                (col('close') - col('prev_close')) / col('prev_close')
            ).otherwise(F.lit(None))
        )
    )

    spark_daily = spark_daily.filter(col('return').isNotNull())

    # Convert to Pandas
    volume_volatility_df = spark_daily.select(
        'base_currency', 'quote_currency', 'date',
        'volume', 'return'
    ).toPandas()

    # Randomly select 1 quote currency
    quote_currencies = volume_volatility_df['quote_currency'].unique()
    selected_quote = np.random.choice(quote_currencies)

    # Then choose 3 bases that share that quote
    base_currencies = volume_volatility_df[
        volume_volatility_df['quote_currency'] == selected_quote
    ]['base_currency'].unique()

    selected_bases = np.random.choice(
        base_currencies,
        size=min(3, len(base_currencies)),
        replace=False
    )

    fig, axes = plt.subplots(3, 2, figsize=(15, 12))

    # Consistent axes across all subplots
    max_volume = volume_volatility_df['volume'].max()
    min_volume = volume_volatility_df['volume'].min()

    max_return = volume_volatility_df['return'].max()
    min_return = volume_volatility_df['return'].min()

    for i, base in enumerate(selected_bases):
        data = volume_volatility_df[
            (volume_volatility_df['base_currency'] == base) &
            (volume_volatility_df['quote_currency'] == selected_quote)
        ]

        returns = data['return']
        volumes = data['volume']

        scatter_ax = axes[i, 0]
        hexbin_ax = axes[i, 1]

        # --- SCATTER PLOT ---
        scatter_ax.scatter(volumes, returns, alpha=0.6, s=10)
        scatter_ax.set_xlabel('Daily Volume')
        scatter_ax.set_ylabel('Daily Return')
        scatter_ax.set_title(f'{base}-{selected_quote}\nScatter Plot')
        scatter_ax.grid(True, alpha=0.3)
        # scatter_ax.set_xlim(min_volume, max_volume)
        scatter_ax.set_ylim(-0.5, 0.5)

        # --- HEXBIN PLOT ---
        hexbin_ax.hexbin(volumes, returns, gridsize=30, cmap='Blues')
        hexbin_ax.set_xlabel('Daily Volume')
        hexbin_ax.set_ylabel('Daily Return')
        hexbin_ax.set_title(f'{base}-{selected_quote}\nHexbin Plot')
        hexbin_ax.grid(True, alpha=0.3)
        # hexbin_ax.set_xlim(min_volume, max_volume)
        hexbin_ax.set_ylim(-0.5, 0.5)

        # Compute Spearman correlation
        correlation, p_value = spearmanr(volumes, returns)

        scatter_ax.text(
            0.05, 0.95,
            f'Spearman r: {correlation:.3f}\np-value: {p_value:.3f}',
            transform=scatter_ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        )

        hexbin_ax.text(
            0.05, 0.95,
            f'Spearman r: {correlation:.3f}\np-value: {p_value:.3f}',
            transform=hexbin_ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        )

    plt.suptitle(
        f'Volume–Return Relationship (Quote: {selected_quote})',
        fontsize=16,
        fontweight='bold'
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    output_path = os.path.join(output_dir, 'volume_return_relationship.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"Figure saved to: {output_path}")

    return volume_volatility_df


# Run
volume_volatility_data = analyze_volume_volatility(spark_df)