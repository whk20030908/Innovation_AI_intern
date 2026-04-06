import pandas as pd
import numpy as np

score = pd.read_csv("../../data/processed/week2_score.csv", index_col=0, parse_dates=True)
rank = pd.read_csv("../../data/processed/week2_sector_rank.csv", index_col=0, parse_dates=True)
drawdown = pd.read_csv("../../data/processed/week2_drawdown.csv", index_col=0, parse_dates=True)
volatility = pd.read_csv("../../data/processed/week2_volatility.csv", index_col=0, parse_dates=True)

# Remove SPY
if "SPY" in drawdown.columns:
    drawdown = drawdown.drop(columns=["SPY"])

if "SPY" in volatility.columns:
    volatility = volatility.drop(columns=["SPY"])

# Align tables
common_cols = score.columns.tolist()
rank = rank[common_cols]
drawdown = drawdown[common_cols]
volatility = volatility[common_cols]

# Convert daily data into month-end data
monthly_score = score.resample("ME").last()
monthly_rank = rank.resample("ME").last()
monthly_drawdown = drawdown.resample("ME").last()
monthly_volatility = volatility.resample("ME").last()

# Drawdown filter
drawdown_filter = monthly_drawdown >= -0.15

# Volatility filter
vol_median = monthly_volatility.median(axis=1)
volatility_filter = monthly_volatility.le(vol_median, axis=0)

# Combine risk filters
combined_filter = drawdown_filter & volatility_filter

# only eligible sector scores
filtered_score = monthly_score.where(combined_filter)

# top 3 sectors
top_sectors = filtered_score.apply(
    lambda row: row.nlargest(3).dropna().index.tolist(),
    axis=1)

# Build equal-weight allocations Assign 1/3
allocations = top_sectors.apply(
    lambda sectors: {s: 1 / 3 for s in sectors} if len(sectors) == 3 else {})

# Convert allocation dictionaries into a structured allocation table
allocation_df = pd.DataFrame(0.0, index=allocations.index, columns=monthly_score.columns)

for date, allocation in allocations.items():
    for sector, weight in allocation.items():
        allocation_df.loc[date, sector] = weight

# Save  files
monthly_rank.to_csv("../../data/processed/week3_monthly_rank.csv")
filtered_score.to_csv("../../data/processed/week3_filtered_score.csv")
allocation_df.to_csv("../../data/processed/week3_allocation_df.csv")

# Save top sectors
top_sectors_df = top_sectors.apply(pd.Series)
top_sectors_df.columns = ["Top_1", "Top_2", "Top_3"]
top_sectors_df.to_csv("../../data/processed/week3_top_sectors.csv")

#Summary
print("Week 3 portfolio construction completed.")
print("Saved files:")
print("- ../../data/processed/week3_monthly_rank.csv")
print("- ../../data/processed/week3_filtered_score.csv")
print("- ../../data/processed/week3_top_sectors.csv")
print("- ../../data/processed/week3_allocation_df.csv")

print("\nSample top sectors:")
print(top_sectors.head())

print("\nSample allocations:")
print(allocation_df.head())