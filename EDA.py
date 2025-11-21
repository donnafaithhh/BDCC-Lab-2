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

# set randomstate
rng = np.random.RandomState(42)

# select 3 base currencies
select_crypto = rng.choice(cryptos)
select_sc = rng.choice(stablecoins)
select_fiats = rng.choice(fiats)

# filtering the dataframe
select_bases = [
    select_crypto, select_sc, select_fiats
]
filtered_df = df.filter(
    df.base_currency.isin(select_bases)
)

# plotting
fig, ax = plt.subplots(
    3, 1, figsize=(15, 5 * len(select_bases))
)
fig.suptitle(
    f'Closing Price Distribution for Base Currencies \
    per Quote Currency'
)

for i, base_currency in enumerate(select_bases):
    base_df = filtered_df.filter(
        filtered_df.base_currency == base_currency
    ).toPandas()
    
    sns.swarmplot(
        data=base_df,
        x='quote_currency',
        y='close',
        ax=ax[i],
        palette='viridis',
        size=2
    )
    ax[i].set_yscale('log')
    ax[i].set_title(f'Base Currency: {base_currency}')
    ax[i].set_xlabel('Quote Currency')
    ax[i].set_ylabel(f'Closing Price (Log Scale)')
    ax[i].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(
    'Closing Price Distribution for Base Currencies per Quote Currency.png'
)
plt.show()

# select 3 base currencies
select_quotes_df = df.select('quote_currency') \
    .distinct() \
    .sample(withReplacement=False, fraction=1.0, seed=42) \
    .limit(3) \
    .collect()
select_quotes = [row['quote_currency'] for row in select_quotes_df]

# filtering the dataframe
filtered_df = df.filter(
    df.quote_currency.isin(select_quotes)
)

# plotting
fig, ax = plt.subplots(
    3, 1, figsize=(15, 5 * len(select_quotes))
)
fig.suptitle(
    f'Closing Price Distribution for Quote Currencies \
    per Base Currency'
)

for i, quote_currency in enumerate(select_quotes):
    quote_df = filtered_df.filter(
        filtered_df.quote_currency == quote_currency
    ).toPandas()
    
    sns.swarmplot(
        data=quote_df,
        x='base_currency',
        y='close',
        ax=ax[i],
        palette='viridis',
        size=2
    )
    ax[i].set_yscale('log')
    ax[i].set_title(f'Quote Currency: {quote_currency}')
    ax[i].set_xlabel('Base Currency')
    ax[i].set_ylabel(f'Closing Price (Log Scale)')
    ax[i].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(
    'Closing Price Distribution for Quote Currencies per Base Currency.png'
)
plt.show()
