import pandas as pd
import numpy as np
from pathlib import Path



prices = pd.read_csv("../../data/raw/etf_prices.csv")
prices["Date"] = pd.to_datetime(prices["Date"])
prices = prices.set_index("Date")

returns = pd.read_csv("../../data/raw/etf_returns.csv")
returns["Date"] = pd.to_datetime(returns["Date"])
returns = returns.set_index("Date")

# Clean data
prices_clean = prices.dropna()
returns_clean = prices_clean.pct_change().dropna()

# risk indicator
momentum = prices_clean.pct_change(63)
volatility = returns_clean.rolling(63).std()

cum = (1 + returns_clean).cumprod()
rolling_max = cum.cummax()
drawdown = (cum - rolling_max) / rolling_max

# Align dates
common_index = (momentum.index.intersection(volatility.index)
    .intersection(drawdown.index))

momentum_aligned = momentum.loc[common_index]
volatility_aligned = volatility.loc[common_index]
drawdown_aligned = drawdown.loc[common_index]

# Risk-adjusted score
score = momentum_aligned - volatility_aligned + drawdown_aligned

# Remove benchmark
sector_score = score.drop(columns=["SPY"])
sector_rank = sector_score.rank(axis=1, ascending=False)


# Use the average of the sector_score to determine the overall ranking
overall_score_rank = sector_score.mean().sort_values(ascending=False)

print("Overall ranking based on average score:")
print(overall_score_rank)

# Save
output_dir = Path("../../data/processed")
output_dir.mkdir(parents=True, exist_ok=True)

prices_clean.to_csv(output_dir / "week2_prices_clean.csv")
returns_clean.to_csv(output_dir / "week2_returns_clean.csv")
momentum_aligned.to_csv(output_dir / "week2_momentum.csv")
volatility_aligned.to_csv(output_dir / "week2_volatility.csv")
drawdown_aligned.to_csv(output_dir / "week2_drawdown.csv")
sector_score.to_csv(output_dir / "week2_score.csv")
sector_rank.to_csv(output_dir / "week2_sector_rank.csv")
overall_score_rank.to_csv(output_dir / "week2_overall_score_rank.csv")

# summary
print("Week 2 analysis completed.\n")

print("Saved files:")
print("- week2_prices_clean.csv")
print("- week2_returns_clean.csv")
print("- week2_momentum.csv")
print("- week2_volatility.csv")
print("- week2_drawdown.csv")
print("- week2_score.csv")
print("- week2_sector_rank.csv\n")

latest_rank = sector_rank.iloc[-1].sort_values()
print("Latest sector ranking:")
print(latest_rank)

