# standard imports
import matplotlib.pyplot as plt
import pandas as pd
import glob
import os

# spark imports
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

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

# Show the result
print("Final DataFrame Schema:")
df.printSchema()
df.cache()

# Save the filtered data just in case
df.write.parquet(
    "Files/top_bases_closing_prices.parquet"
)

# Create a summary statistics table
print(f"\nSummary statistics for closing prices:")
df.select(
    F.mean("close").alias("mean_close"),
    F.stddev("close").alias("std_close"),
    F.min("close").alias("min_close"),
    F.max("close").alias("max_close")
).show()

