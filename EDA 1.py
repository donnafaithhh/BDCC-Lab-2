# standard imports
import matplotlib.pyplot as plt
import random
import pandas as pd
import numpy as np
import glob
import os

# spark imports
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import FloatType
import pyspark.sql.window as window

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
top_bases = counts_df.nlargest(75, 'quote_count')

# Initialize Spark Session
spark = SparkSession.builder.appName("TopBasesAnalysis").getOrCreate()

# Directory containing the parquet files
directory = '/mnt/data/public/binance-full-history'
files = [f for f in os.listdir(directory) if f.endswith('.parquet')]

# Read the main Binance data
binance = spark.read.parquet(directory)

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
    pair = f.replace('.parquet','')
    base, quote = pair.split('-')
    
    if base in cryptos:
        classified["cryptocurrency"].append(f)
    elif base in stablecoins:
        classified["stablecoin"].append(f)
    elif base in fiats:
        classified["fiat_backed"].append(f)
    else:
        classified["cryptocurrency"].append(f)

classification = {}
for category, file_list in classified.items():
    for f in file_list:
        pair = f.replace(".parquet", "")
        classification[pair] = category

# Add stock column and create classification DataFrame
binance_with_stock = binance.withColumn(
    "stock",
    F.regexp_extract(F.input_file_name(), r'([^/]+)\.parquet', 1)
)

class_rows = [
    (pair, category)
    for pair, category in classification.items()
]

class_df = spark.createDataFrame(class_rows, ["stock", "classification"])

# Create the main DataFrame with classifications
df = (
    binance_with_stock
    .join(class_df, on="stock", how="left")
    .withColumn("base_currency", F.split(F.col("stock"), "-")[0])
    .withColumn("quote_currency", F.split(F.col("stock"), "-")[1])
)

# Final DataFrame with only top base currencies and closing price
top_base_list = top_bases['base_currency'].to_list()
df = (
    df
    .filter(F.col("base_currency").isin(top_base_list))
    .select(
        "stock",
        "base_currency",
        "quote_currency", 
        "open_time",
        "close",
        "classification"
    )
)

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# 1. Compare same quote currency across different base currencies
print("Analyzing quote currencies across base currencies...")

# First, find common quote currencies that exist for all top base currencies
quote_currency_counts = (
    df
    .select("base_currency", "quote_currency")
    .distinct()
    .groupBy("quote_currency")
    .agg(F.count("base_currency").alias("base_count"))
)

total_base_currencies = len(top_base_list)
common_quote_currencies = (
    quote_currency_counts
    .filter(F.col("base_count") == total_base_currencies)
    .select("quote_currency")
    .collect()
)

common_quote_list = [row.quote_currency for row in common_quote_currencies]
print(f"Found {len(common_quote_list)} common quote currencies across all base currencies")

# Randomly select 3 quote currencies from common ones
if len(common_quote_list) >= 3:
    selected_quotes = random.sample(common_quote_list, 3)
else:
    selected_quotes = common_quote_list

print(f"Selected quote currencies: {selected_quotes}")

# Create boxplots for each selected quote currency across base currencies
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
if len(selected_quotes) < 3:
    fig, axes = plt.subplots(1, len(selected_quotes), figsize=(6*len(selected_quotes), 6))
    if len(selected_quotes) == 1:
        axes = [axes]

for i, quote_currency in enumerate(selected_quotes):
    # Filter data for this quote currency and sample for visualization
    quote_data = (
        df
        .filter(F.col("quote_currency") == quote_currency)
        .select("base_currency", "close")
        .sample(fraction=0.1)  # Sample 10% for visualization
        .toPandas()
    )
    
    # Group by base_currency and get closing prices
    boxplot_data = []
    base_labels = []
    
    for base_currency in top_base_list[:15]:  # Limit to top 15 for readability
        base_prices = quote_data[quote_data['base_currency'] == base_currency]['close'].values
        if len(base_prices) > 0:
            boxplot_data.append(base_prices)
            base_labels.append(base_currency)
    
    if boxplot_data:
        axes[i].boxplot(boxplot_data, labels=base_labels, showfliers=False)
        axes[i].set_title(f'Closing Prices - Quote: {quote_currency}')
        axes[i].set_xlabel('Base Currency')
        axes[i].set_ylabel('Closing Price')
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.suptitle('Closing Price Distribution: Same Quote Currency Across Different Base Currencies', 
             y=1.02, fontsize=14)
plt.savefig(
    'Figures/Closing Price, Quote Currency.png'
)
plt.show()

# 2. Compare different quote currencies across the same base currency
print("\nAnalyzing different quote currencies across base currencies...")

# Randomly select 3 base currencies from top bases
selected_bases = random.sample(top_base_list, 3)
print(f"Selected base currencies: {selected_bases}")

# Create boxplots for each selected base currency across quote currencies
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for i, base_currency in enumerate(selected_bases):
    # Get all quote currencies for this base
    base_quotes = (
        df
        .filter(F.col("base_currency") == base_currency)
        .select("quote_currency")
        .distinct()
        .collect()
    )
    
    quote_list = [row.quote_currency for row in base_quotes]
    
    # Sample a subset of quote currencies for readability (max 15)
    if len(quote_list) > 15:
        display_quotes = random.sample(quote_list, 15)
    else:
        display_quotes = quote_list
    
    # Filter data for this base currency and sample for visualization
    base_data = (
        df
        .filter((F.col("base_currency") == base_currency) & 
                (F.col("quote_currency").isin(display_quotes)))
        .select("quote_currency", "close")
        .sample(fraction=0.1)  # Sample 10% for visualization
        .toPandas()
    )
    
    # Group by quote_currency and get closing prices
    boxplot_data = []
    quote_labels = []
    
    for quote_currency in display_quotes:
        quote_prices = base_data[base_data['quote_currency'] == quote_currency]['close'].values
        if len(quote_prices) > 0:
            boxplot_data.append(quote_prices)
            quote_labels.append(quote_currency)
    
    if boxplot_data:
        axes[i].boxplot(boxplot_data, labels=quote_labels, showfliers=False)
        axes[i].set_title(f'Closing Prices - Base: {base_currency}')
        axes[i].set_xlabel('Quote Currency')
        axes[i].set_ylabel('Closing Price')
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.suptitle('Closing Price Distribution: Different Quote Currencies Across Same Base Currency', 
             y=1.02, fontsize=14)
plt.savefig(
    'Figures/Closing Price, Base Currency.png'
)
plt.show()



