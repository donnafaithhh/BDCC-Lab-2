# typical
import os
import re
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

# pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import explode, desc
from pyspark.sql.functions import col, lag
from pyspark.sql import functions as F
from pyspark.sql.window import Window
import time

spark = (SparkSession
     .builder
     .master('local[*]')
     .getOrCreate())

directory = '/mnt/data/public/binance-full-history'
binance = spark.read.parquet(directory)
files = os.listdir(directory)

# Base Classifications (cryptocurrencies, stablecoins, and fiats)
cryptos = [
    '1INCH', 'AAVE', 'ACM', 'ADA', 'ADADOWN', 'ADAUP', 'ADX', 'AE', 'AERGO', 'AGI',
    'AION', 'AKRO', 'ALGO', 'ALICE', 'ALPHA', 'AMB', 'ANKR', 'ANT', 'APPC', 'AR',
    'ARDR', 'ARK', 'ARN', 'ARPA', 'ASR', 'AST', 'ATA', 'ATM', 'ATOM', 'AUCTION',
    'AUDIO', 'BCH', 'BEAM', 'BEL', 'BETA', 'BETH', 'BIFI', 'BLZ', 'BNB', 'BNBDOWN',
    'BNBUP', 'BNT', 'BQX', 'BRD', 'BTC', 'BTCDOWN', 'BTCST', 'BTCUP', 'BTG', 'BTS',
    'BTT', 'BURGER', 'BZRX', 'C98', 'CAKE', 'CTK', 'CTSI', 'CTXC', 'CVC', 'CVP',
    'DAR', 'DASH', 'DATA', 'DCR', 'DEGO', 'DENT', 'DEXE', 'DF', 'DGB', 'DGD', 'DIA',
    'DLT', 'DNT', 'DOCK', 'DODO', 'DOGE', 'DOT', 'DOTDOWN', 'DOTUP', 'DREP', 'DUSK',
    'DYDX', 'EDO', 'EGLD', 'ELF', 'ENG', 'ENJ', 'EOS', 'ETC', 'ETH', 'FET', 'FIL',
    'FIO', 'FIRO', 'FIS', 'FLM', 'FLOW', 'FOR', 'FORTH', 'FRONT', 'FTM', 'FTT',
    'FUEL', 'FUN', 'FXS', 'GALA', 'GAS', 'GHST', 'GLM', 'GNT', 'GO', 'GRS', 'GRT',
    'GTC', 'GTO', 'GVT', 'GXS', 'HARD', 'HBAR', 'HC', 'HIVE', 'HNT', 'HOT', 'ICP',
    'ICX', 'IOTA', 'KSM', 'LAZIO', 'LEND', 'LINA', 'LINK', 'LINKDOWN', 'LINKUP',
    'LIT', 'LOOM', 'LPT', 'LRC', 'LSK', 'LTC', 'LTCDOWN', 'LTCUP', 'LTO', 'LUN',
    'LUNA', 'MANA', 'MASK', 'MATIC', 'MBL', 'MBOX', 'MCO', 'MDA', 'MDT', 'MDX',
    'MFT', 'MINA', 'MIR', 'MITH', 'NEO', 'OCEAN', 'OG', 'OGN', 'OM', 'OMG', 'ONE',
    'ONG', 'ONT', 'ORN', 'OST', 'OXT', 'PAXG', 'PERL', 'PERP', 'PHA', 'PHB', 'PIVX',
    'PNT', 'POA', 'POE', 'POLS', 'POLY', 'POND', 'POWR', 'PPT', 'PROM', 'PROS',
    'PSG', 'PUNDIX', 'QKC', 'QLC', 'QNT', 'QSP', 'QTUM', 'RAMP', 'RCN', 'RDN',
    'REEF', 'REN', 'REP', 'REQ', 'SNGLS', 'SNM', 'SNT', 'SNX', 'SOL', 'SPARTA',
    'SRM', 'STEEM', 'STMX', 'STORJ', 'STORM', 'STPT', 'STRAT', 'STRAX', 'STX',
    'SUN', 'SUPER', 'SUSHI', 'SUSHIDOWN', 'SUSHIUP', 'SXP', 'SXPUP', 'SYS', 'TCT',
    'TFUEL', 'THETA', 'TKO', 'TLM', 'TNB', 'TNT', 'TOMO', 'TORN', 'TRB', 'TROY',
    'TRX', 'VET', 'VIA', 'VIB', 'VIBE', 'VIDT', 'VITE', 'VTHO', 'WABI', 'WAN',
    'WAVES', 'WBTC', 'WIN', 'WING', 'WNXM', 'WPR', 'WRX', 'WTC', 'XEC', 'XEM',
    'XLM', 'XMR', 'XRP', 'XRPDOWN', 'XRPUP', 'XTZ']
stablecoins = ['USDT','USDC','BUSD','TUSD','DAI','PAX','BIDR','IDRT']
fiats = ['EUR','GBP','AUD','TRY','BRL','RUB','NGN','UAH']

classified = {
    "cryptocurrency": [],
    "stablecoin": [],
    "fiat_backed": []
}

for f in files:
    if f.endswith('.parquet'):
        pair = f.replace('.parquet','')
        base, quote = pair.split('-')
        
        if base in cryptos:
            classified["cryptocurrency"].append(f)
        elif base in stablecoins:
            classified["stablecoin"].append(f)
        elif base in fiats:
            classified["fiat_backed"].append(f)
        else:
            # unlisted base asset
            classified["cryptocurrency"].append(f)

classification = {}

for category, file_list in classified.items():
    for f in file_list:
        pair = f.replace(".parquet", "")
        classification[pair] = category

binance_with_stock = binance.withColumn(
    "stock",
    F.regexp_extract(F.input_file_name(), r'([^/]+)\.parquet', 1)
)

class_rows = [
    (pair, category)
    for pair, category in classification.items()
]

class_df = spark.createDataFrame(class_rows, ["stock", "classification"])

df = (
    binance_with_stock
    .join(class_df, on="stock", how="left")
    .withColumn("base_currency", F.split(F.col("stock"), "-")[0])
    .withColumn("quote_currency", F.split(F.col("stock"), "-")[1])
)

# FIX: Calculate log returns before computing volatility
df_with_returns = df.withColumn(
    "log_return", 
    F.log(F.col("close") / F.lag("close").over(
        Window.partitionBy("stock").orderBy("open_time")
    ))
)

def compute_volatility_metrics(df):
    """
    Compute volatility metrics using Spark's built-in parallel processing
    """
    # Extract year from open_time
    df_with_year = df.withColumn("year", F.year("open_time"))
    
    # Filter to top 50 base currencies by data volume for meaningful analysis
    top_currencies = df_with_year.groupBy("base_currency") \
        .count() \
        .orderBy(F.desc("count")) \
        .limit(50) \
        .select("base_currency") \
        .rdd.flatMap(lambda x: x).collect()
    
    df_filtered = df_with_year.filter(F.col("base_currency").isin(top_currencies))
    
    # Calculate volatility metrics per base_currency, year, and classification
    volatility_df = df_filtered.groupBy("base_currency", "year", "classification") \
        .agg(
            F.stddev("log_return").alias("volatility_std"),
            F.mean("log_return").alias("mean_return"),
            F.count("log_return").alias("n_observations")
        ) \
        .filter(F.col("n_observations") > 10)  # Filter out currencies with insufficient data
    
    return volatility_df

def compute_year_volatility(df):
    """
    Compute annual volatility across all base currencies
    """
    df_with_year = df.withColumn("year", F.year("open_time"))
    
    # Filter to top 50 base currencies for consistency
    top_currencies = df_with_year.groupBy("base_currency") \
        .count() \
        .orderBy(F.desc("count")) \
        .limit(50) \
        .select("base_currency") \
        .rdd.flatMap(lambda x: x).collect()
    
    df_filtered = df_with_year.filter(F.col("base_currency").isin(top_currencies))
    
    # Calculate annual volatility distribution per year
    annual_volatility = df_filtered.groupBy("base_currency", "year") \
        .agg(F.stddev("log_return").alias("annual_volatility")) \
        .filter(F.col("annual_volatility").isNotNull())
    
    return annual_volatility

def create_volatility_plots(volatility_pdf, annual_vol_pdf):
    """
    Create visualization plots using the computed metrics
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    # Plot 1: Volatility comparison by classification
    sns.boxplot(data=volatility_pdf, x='classification', y='volatility_std', ax=ax1)
    ax1.set_title('Volatility Distribution by Currency Classification')
    ax1.set_ylabel('Standard Deviation of Log Returns')
    ax1.set_xlabel('Currency Classification')
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot 2: Year-to-year market volatility changes
    sns.boxplot(data=annual_vol_pdf, x='year', y='annual_volatility', ax=ax2)
    ax2.set_title('Year-to-Year Market Volatility Changes (Top 50 Base Currencies)')
    ax2.set_ylabel('Annual Volatility (Std of Log Returns)')
    ax2.set_xlabel('Year')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('volatility_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print("Volatility Summary by Classification:")
    print(volatility_pdf.groupby('classification')['volatility_std'].describe())
    
    print("\nAnnual Volatility Summary:")
    print(annual_vol_pdf.groupby('year')['annual_volatility'].describe())

# Main execution
if __name__ == "__main__":
    # Compute metrics using Spark's parallel processing
    print("Computing volatility metrics...")
    volatility_metrics_df = compute_volatility_metrics(df_with_returns)
    annual_volatility_df = compute_year_volatility(df_with_returns)
    
    # Collect results to Pandas for plotting
    print("Collecting results...")
    volatility_pandas_df = volatility_metrics_df.toPandas()
    annual_vol_pandas_df = annual_volatility_df.toPandas()
    
    print(f"Volatility metrics computed for {len(volatility_pandas_df)} currency-year-classification combinations")
    print(f"Annual volatility computed for {len(annual_vol_pandas_df)} currency-year pairs")
    
    # Create visualizations
    print("Creating plots...")
    create_volatility_plots(volatility_pandas_df, annual_vol_pandas_df)
    
    print("Analysis complete!")