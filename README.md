# Binance for Beginners

## 📋 Repository Overview
The works in this repository were created by Donna Faith Go and Raymond Castro for our Big Data and Cloud Computing class. 
Here, we applied what we know about big data on the Binance Full History dataset where we took the perspective of first-time investors, new to trading cryptocurrency.
> 📄 **Full project details:** [final report.html](final report.html)  

## 📋 Project Abstract
Cryptocurrency trading has grown in popularity recently, yet it remains challenging for first-time investors to learn due to the large number of assets and quote currencies available on platforms like Binance. 
This study investigates how volatility, price stability, and trading activity patterns differ across base cryptocurrencies and their quote currencies to provide data-driven guidance for beginners. 
Using historical Binance data from 2017 to 2022, we created a Spark DataFrame to handle over 30 Parquet files totaling 33 GB. 
The dataset included trading information such as prices, volumes, and number of trades. 
Exploratory data analysis focused on base currencies with at least eight quote currencies to enable meaningful comparisons. 
Correlation analyses examined how returns of base currencies relate to different quote currencies and to each other. 
Annual volatility was computed from daily log returns to identify periods of high and low price fluctuations, while trading volume patterns were analyzed to assess market activity. 
Results indicate that major cryptocurrencies like BTC and ETH exhibit high return correlations across different quote currencies, suggesting their performance is largely independent of the quote currency and generally stable. 
Less established coins like ADA show lower correlations and higher sensitivity to macroeconomic conditions, indicating greater risk for beginners. 
Volatility tends to spike following the introduction of new quote currencies, while trading volume does not consistently predict returns, reflecting unique market behaviors. 
For first-time traders, the study recommends prioritizing high-liquidity, high-stability pairs such as BTC-USDT or ETH-USDT, monitoring volatility to time trades, and using smaller positions for more volatile or less established coins.
However, it must be noted that when replicating this study, a more recent dataset should be used since we only used data from 2017 until 2022.
For more accurate results, future work should incorporate data from at least a year from present onwards. 
