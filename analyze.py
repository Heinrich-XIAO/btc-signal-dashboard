"""Initial pattern analysis before strategy creation."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (14, 8)
plt.rcParams["font.size"] = 10

df = pd.read_csv("btc_5m_features.csv", index_col="timestamp", parse_dates=True)

# Drop NaNs for analysis
df_clean = df.dropna()
print(f"Data shape after dropping NaNs: {df_clean.shape}")
print(f"Up moves: {df_clean['target'].sum()} ({df_clean['target'].mean()*100:.1f}%)")
print(f"Down moves: {(1-df_clean['target']).sum()} ({(1-df_clean['target']).mean()*100:.1f}%)")

# Correlation analysis with target
corr_with_target = df_clean.corr()["target"].drop("target").sort_values(key=lambda x: np.abs(x), ascending=False)
print("\nTop 20 features by correlation with target:")
print(corr_with_target.head(20))

# Also look at correlations for conditional subsets
print("\n--- Conditional Analysis ---")

# After an up candle, what happens next?
up_then = df_clean[df_clean["returns_lag1"] > 0]
down_then = df_clean[df_clean["returns_lag1"] < 0]
print(f"After UP candle: next up = {up_then['target'].mean()*100:.1f}%")
print(f"After DOWN candle: next up = {down_then['target'].mean()*100:.1f}%")

# After large moves
large_up = df_clean[df_clean["returns_lag1"] > df_clean["returns_lag1"].quantile(0.9)]
large_down = df_clean[df_clean["returns_lag1"] < df_clean["returns_lag1"].quantile(0.1)]
print(f"After LARGE UP (top 10%): next up = {large_up['target'].mean()*100:.1f}%")
print(f"After LARGE DOWN (bottom 10%): next up = {large_down['target'].mean()*100:.1f}%")

# RSI extremes
rsi_oversold = df_clean[df_clean["rsi_14"] < 30]
rsi_overbought = df_clean[df_clean["rsi_14"] > 70]
print(f"RSI < 30 (oversold): next up = {rsi_oversold['target'].mean()*100:.1f}% (n={len(rsi_oversold)})")
print(f"RSI > 70 (overbought): next up = {rsi_overbought['target'].mean()*100:.1f}% (n={len(rsi_overbought)})")

# Bollinger Band position
bb_low = df_clean[df_clean["bb_20_position"] < 0.1]
bb_high = df_clean[df_clean["bb_20_position"] > 0.9]
print(f"BB position < 0.1: next up = {bb_low['target'].mean()*100:.1f}% (n={len(bb_low)})")
print(f"BB position > 0.9: next up = {bb_high['target'].mean()*100:.1f}% (n={len(bb_high)})")

# Volume spikes
vol_spike = df_clean[df_clean["volume_ratio"] > 2]
print(f"Volume spike (>2x avg): next up = {vol_spike['target'].mean()*100:.1f}% (n={len(vol_spike)})")

# Stochastic extremes
stoch_low = df_clean[df_clean["stoch_k_14"] < 20]
stoch_high = df_clean[df_clean["stoch_k_14"] > 80]
print(f"Stoch < 20: next up = {stoch_low['target'].mean()*100:.1f}% (n={len(stoch_low)})")
print(f"Stoch > 80: next up = {stoch_high['target'].mean()*100:.1f}% (n={len(stoch_high)})")

# Time of day
for hour in range(24):
    hour_df = df_clean[df_clean["hour"] == hour]
    if len(hour_df) > 100:
        print(f"Hour {hour:02d}: next up = {hour_df['target'].mean()*100:.1f}% (n={len(hour_df)})")

# --- Graphs ---

# 1. Feature correlation bar plot
fig, ax = plt.subplots(figsize=(12, 8))
top_corr = corr_with_target.head(25)
colors = ["green" if x > 0 else "red" for x in top_corr.values]
top_corr.plot(kind="barh", color=colors, ax=ax)
ax.set_title("Top 25 Features by Correlation with Next 5m Direction", fontsize=14, fontweight="bold")
ax.set_xlabel("Correlation with Target (Up=1, Down=0)")
ax.axvline(x=0, color="black", linewidth=0.8)
plt.tight_layout()
plt.savefig("graphs/01_feature_correlations.png", dpi=150, bbox_inches="tight")
plt.close()

# 2. RSI distribution by target
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
df_clean[df_clean["target"] == 1]["rsi_14"].hist(bins=50, alpha=0.7, label="Next Up", ax=axes[0], color="green")
df_clean[df_clean["target"] == 0]["rsi_14"].hist(bins=50, alpha=0.7, label="Next Down", ax=axes[0], color="red")
axes[0].set_title("RSI(14) Distribution by Next Move")
axes[0].legend()

# BB position distribution
df_clean[df_clean["target"] == 1]["bb_20_position"].hist(bins=50, alpha=0.7, label="Next Up", ax=axes[1], color="green")
df_clean[df_clean["target"] == 0]["bb_20_position"].hist(bins=50, alpha=0.7, label="Next Down", ax=axes[1], color="red")
axes[1].set_title("BB(20) Position Distribution by Next Move")
axes[1].legend()
plt.tight_layout()
plt.savefig("graphs/02_rsi_bb_distributions.png", dpi=150, bbox_inches="tight")
plt.close()

# 3. After-move analysis
fig, ax = plt.subplots(figsize=(10, 6))
lag1_bins = pd.qcut(df_clean["returns_lag1"], q=20, duplicates="drop")
lag1_analysis = df_clean.groupby(lag1_bins)["target"].mean()
lag1_counts = df_clean.groupby(lag1_bins)["target"].count()

x_pos = range(len(lag1_analysis))
ax.bar(x_pos, lag1_analysis.values, color=["green" if x > 0.5 else "red" for x in lag1_analysis.values])
ax.axhline(y=0.5, color="black", linestyle="--", label="50% baseline")
ax.set_title("Next Candle Up Probability by Previous Candle Return (20 Quantiles)", fontsize=12, fontweight="bold")
ax.set_xlabel("Previous Candle Return Quantile (Low → High)")
ax.set_ylabel("Probability Next Candle is Up")
plt.tight_layout()
plt.savefig("graphs/03_after_move_analysis.png", dpi=150, bbox_inches="tight")
plt.close()

# 4. Hour-of-day pattern
fig, ax = plt.subplots(figsize=(12, 5))
hour_stats = df_clean.groupby("hour")["target"].agg(["mean", "count"])
hour_stats = hour_stats[hour_stats["count"] > 100]
ax.bar(hour_stats.index, hour_stats["mean"], color=["green" if x > 0.5 else "red" for x in hour_stats["mean"]])
ax.axhline(y=0.5, color="black", linestyle="--")
ax.set_title("Up Probability by Hour of Day (UTC)", fontsize=12, fontweight="bold")
ax.set_xlabel("Hour (UTC)")
ax.set_ylabel("Probability Next Candle is Up")
plt.tight_layout()
plt.savefig("graphs/04_hour_of_day_pattern.png", dpi=150, bbox_inches="tight")
plt.close()

# 5. Volume spike analysis
fig, ax = plt.subplots(figsize=(10, 6))
vol_bins = pd.qcut(df_clean["volume_ratio"], q=20, duplicates="drop")
vol_analysis = df_clean.groupby(vol_bins)["target"].mean()
ax.bar(range(len(vol_analysis)), vol_analysis.values, color=["green" if x > 0.5 else "red" for x in vol_analysis.values])
ax.axhline(y=0.5, color="black", linestyle="--")
ax.set_title("Next Candle Up Probability by Volume Ratio (20 Quantiles)", fontsize=12, fontweight="bold")
ax.set_xlabel("Volume Ratio Quantile (Low → High)")
ax.set_ylabel("Probability Next Candle is Up")
plt.tight_layout()
plt.savefig("graphs/05_volume_analysis.png", dpi=150, bbox_inches="tight")
plt.close()

# 6. Momentum (cumulative return) analysis
fig, ax = plt.subplots(figsize=(10, 6))
cumret_bins = pd.qcut(df_clean["cumret_5"], q=20, duplicates="drop")
cumret_analysis = df_clean.groupby(cumret_bins)["target"].mean()
ax.bar(range(len(cumret_analysis)), cumret_analysis.values, color=["green" if x > 0.5 else "red" for x in cumret_analysis.values])
ax.axhline(y=0.5, color="black", linestyle="--")
ax.set_title("Next Candle Up Probability by 5-Candle Cumulative Return (20 Quantiles)", fontsize=12, fontweight="bold")
ax.set_xlabel("5-Candle Cum Return Quantile (Low → High)")
ax.set_ylabel("Probability Next Candle is Up")
plt.tight_layout()
plt.savefig("graphs/06_momentum_analysis.png", dpi=150, bbox_inches="tight")
plt.close()

print("\nAnalysis graphs saved to graphs/ directory.")
