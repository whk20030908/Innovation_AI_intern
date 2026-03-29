import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# 1. Define ETFs and benchmark
etfs = ['XLK', 'XLF', 'XLE', 'XLV', 'XLY', 'XLI', 'XLP', 'XLU', 'XLB', 'XLRE', 'XLC']
benchmark = 'SPY'

# 2. Combine into one list
tickers = etfs + [benchmark]
print("\nTickers:")
print(tickers)

# 3. Download historical data
data = yf.download(tickers, start="2015-01-01", end="2025-01-01")

# 4. Check
print("\nPrice fields:")
print(data.columns.get_level_values(0).unique())

# 5. Extract closing prices
prices = data['Close']
print("\nClosing price table:")
print(prices.head())

# 6. Save price data
prices.to_csv('etf_prices.csv')
print("\nPrice data saved")

# 7. daily returns
returns = prices.pct_change().dropna()
print("\nDaily returns:")
print(returns.head())

# 8. Save returns
returns.to_csv('etf_returns.csv')
print("\nReturn data Saved")

# 9. volatility
volatility = returns.std()
print("\nDaily volatility:")
print(volatility)

# 10. Correlation matrix
correlation = returns.corr()
print("\nCorrelation matrix:")
print(correlation)

# 11. correlation with SPY only
spy_corr = correlation['SPY'].drop('SPY')
print("\nCorrelation(SPY):")
print(spy_corr.sort_values(ascending=False))

# 12. Plot daily volatility
plt.figure(figsize=(10, 6))
volatility.sort_values().plot(kind='bar')
plt.title('Daily Volatility of ETFs')
plt.ylabel('Volatility')
plt.xlabel('Ticker')
plt.tight_layout()
plt.show()

# 13. Plot correlation with SPY
plt.figure(figsize=(10, 6))
spy_corr.sort_values().plot(kind='bar')
plt.title('Correlation of ETFs with SPY')
plt.ylabel('Correlation')
plt.xlabel('Ticker')
plt.tight_layout()
plt.show()