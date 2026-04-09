import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

allocation_df = pd.read_csv("../../data/processed/week3_allocation_df.csv", index_col=0, parse_dates=True)
sector_returns = pd.read_csv("../../data/processed/week2_returns_clean.csv", index_col=0, parse_dates=True)
prices = pd.read_csv("../../data/processed/week2_prices_clean.csv", index_col=0, parse_dates=True)

# Separate SPY
spy_returns = sector_returns["SPY"]
sector_returns_only = sector_returns.drop(columns=["SPY"])

# Expand monthly allocation into daily weights
daily_weights = allocation_df.reindex(sector_returns_only.index, method="ffill")

# Shift weights
daily_weights_shifted = daily_weights.shift(1)

# Replace missing values
daily_weights_shifted = daily_weights_shifted.fillna(0)

# strategy daily returns
strategy_daily_returns = (daily_weights_shifted * sector_returns_only).sum(axis=1)

# Simulate strategy portfolio value
initial_value = 100000
portfolio_value = initial_value * (1 + strategy_daily_returns).cumprod()

# strategy performance metrics
strategy_cumulative_return = portfolio_value.iloc[-1] / portfolio_value.iloc[0] - 1

years = len(portfolio_value) / 252
strategy_cagr = (portfolio_value.iloc[-1] / portfolio_value.iloc[0]) ** (1 / years) - 1

strategy_annual_volatility = strategy_daily_returns.std() * np.sqrt(252)

strategy_rolling_max = portfolio_value.cummax()
strategy_drawdown_series = (portfolio_value - strategy_rolling_max) / strategy_rolling_max
strategy_max_drawdown = strategy_drawdown_series.min()

strategy_sharpe_ratio = (strategy_daily_returns.mean() / strategy_daily_returns.std()) * np.sqrt(252)

# SPY
spy_portfolio_value = initial_value * (1 + spy_returns).cumprod()
spy_cumulative_return = spy_portfolio_value.iloc[-1] / spy_portfolio_value.iloc[0] - 1
spy_cagr = (spy_portfolio_value.iloc[-1] / spy_portfolio_value.iloc[0]) ** (1 / years) - 1
spy_annual_volatility = spy_returns.std() * np.sqrt(252)

spy_rolling_max = spy_portfolio_value.cummax()
spy_drawdown_series = (spy_portfolio_value - spy_rolling_max) / spy_rolling_max
spy_max_drawdown = spy_drawdown_series.min()

spy_sharpe_ratio = (spy_returns.mean() / spy_returns.std()) * np.sqrt(252)

# Equal-weight sector portfolio
equal_weight_returns = sector_returns_only.mean(axis=1)
equal_weight_portfolio_value = initial_value * (1 + equal_weight_returns).cumprod()
equal_weight_cumulative_return = equal_weight_portfolio_value.iloc[-1] / equal_weight_portfolio_value.iloc[0] - 1
equal_weight_cagr = (equal_weight_portfolio_value.iloc[-1] / equal_weight_portfolio_value.iloc[0]) ** (1 / years) - 1
equal_weight_annual_volatility = equal_weight_returns.std() * np.sqrt(252)

equal_weight_rolling_max = equal_weight_portfolio_value.cummax()
equal_weight_drawdown_series = (equal_weight_portfolio_value - equal_weight_rolling_max) / equal_weight_rolling_max
equal_weight_max_drawdown = equal_weight_drawdown_series.min()

equal_weight_sharpe_ratio = (equal_weight_returns.mean() / equal_weight_returns.std()) * np.sqrt(252)

# portfolio value series
portfolio_value.to_csv("../../data/processed/week4_strategy_portfolio_value.csv")
spy_portfolio_value.to_csv("../../data/processed/week4_spy_portfolio_value.csv")
equal_weight_portfolio_value.to_csv("../../data/processed/week4_equal_weight_portfolio_value.csv")

# daily return series
strategy_daily_returns.to_csv("../../data/processed/week4_strategy_daily_returns.csv")
spy_returns.to_csv("../../data/processed/week4_spy_daily_returns.csv")
equal_weight_returns.to_csv("../../data/processed/week4_equal_weight_daily_returns.csv")

# Summary table
summary_df = pd.DataFrame({
    "Portfolio": ["Strategy", "SPY", "Equal-Weight Sector Portfolio"],
    "Cumulative Return": [
        strategy_cumulative_return,
        spy_cumulative_return,
        equal_weight_cumulative_return
    ],
    "CAGR": [
        strategy_cagr,
        spy_cagr,
        equal_weight_cagr
    ],
    "Annualized Volatility": [
        strategy_annual_volatility,
        spy_annual_volatility,
        equal_weight_annual_volatility
    ],
    "Maximum Drawdown": [
        strategy_max_drawdown,
        spy_max_drawdown,
        equal_weight_max_drawdown
    ],
    "Sharpe Ratio": [
        strategy_sharpe_ratio,
        spy_sharpe_ratio,
        equal_weight_sharpe_ratio
    ]
})

summary_df.to_csv("../../data/processed/week4_performance_summary.csv", index=False)

# Plot strategy vs benchmarks
plt.figure(figsize=(12, 6))
plt.plot(portfolio_value.index, portfolio_value.values, label="Strategy")
plt.plot(spy_portfolio_value.index, spy_portfolio_value.values, label="SPY")
plt.plot(equal_weight_portfolio_value.index, equal_weight_portfolio_value.values, label="Equal-Weight Sector Portfolio")

plt.title("Strategy vs Benchmarks")
plt.xlabel("Date")
plt.ylabel("Portfolio Value")
plt.legend()
plt.grid(True)

plt.savefig("../../output/figure/Week4_Strategy_vs_Benchmarks.png", dpi=300, bbox_inches="tight")
plt.show()

# Plot drawdown comparison
plt.figure(figsize=(12, 6))
plt.plot(strategy_drawdown_series.index, strategy_drawdown_series.values, label="Strategy")
plt.plot(spy_drawdown_series.index, spy_drawdown_series.values, label="SPY")
plt.plot(equal_weight_drawdown_series.index, equal_weight_drawdown_series.values, label="Equal-Weight Sector Portfolio")

plt.title("Drawdown Comparison")
plt.xlabel("Date")
plt.ylabel("Drawdown")
plt.legend()
plt.grid(True)

plt.savefig("../../output/figure/Week4_Drawdown_Comparison.png", dpi=300, bbox_inches="tight")
plt.show()

# Build volatility comparison table
volatility_df = pd.DataFrame({
    "Portfolio": ["Strategy", "SPY", "Equal-Weight Sector Portfolio"],
    "Annualized Volatility": [
        strategy_annual_volatility,
        spy_annual_volatility,
        equal_weight_annual_volatility
    ]
})

volatility_df.to_csv("../../data/processed/week4_volatility_comparison.csv", index=False)

# Plot volatility comparison
plt.figure(figsize=(10, 6))
plt.bar(volatility_df["Portfolio"], volatility_df["Annualized Volatility"])

plt.title("Annualized Volatility Comparison")
plt.xlabel("Portfolio")
plt.ylabel("Annualized Volatility")
plt.grid(True, axis="y")

plt.savefig("../../output/figure/Week4_Volatility_Comparison.png", dpi=300, bbox_inches="tight")
plt.show()

# Summary
print("Week 4 backtesting completed.")
print("Saved files:")
print("- ../../data/processed/week4_strategy_portfolio_value.csv")
print("- ../../data/processed/week4_spy_portfolio_value.csv")
print("- ../../data/processed/week4_equal_weight_portfolio_value.csv")
print("- ../../data/processed/week4_strategy_daily_returns.csv")
print("- ../../data/processed/week4_spy_daily_returns.csv")
print("- ../../data/processed/week4_equal_weight_daily_returns.csv")
print("- ../../data/processed/week4_performance_summary.csv")
print("- ../../output/figure/Week4_Strategy_vs_Benchmarks.png")

print("\nPerformance Summary:")
print(summary_df)