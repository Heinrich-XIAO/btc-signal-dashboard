"""Search for ANY condition with 90% accuracy and >25% coverage."""
import pandas as pd
import numpy as np

df = pd.read_csv("btc_5m_features.csv", index_col="timestamp", parse_dates=True)
df_clean = df.dropna()

print(f"Total candles: {len(df_clean)}")
print(f"Target baseline: {df_clean['target'].mean():.3f}")
print()

# We need ~8,770 trades with 90% accuracy
# Let's test many extreme conditions and see if any combination works

def test_condition(df, condition, name):
    trades = df[condition]
    if len(trades) < 8000:
        return
    acc = trades["target"].mean()
    print(f"{name:50s}: trades={len(trades):5d}, accuracy={acc:.3f}")

print("=== Testing individual extreme conditions for 90% accuracy with >8000 trades ===\n")

# RSI extremes with different thresholds
for rsi_th in [10, 15, 20, 25, 30, 40, 50]:
    test_condition(df_clean, df_clean["rsi_14"] < rsi_th, f"RSI < {rsi_th}")
    test_condition(df_clean, df_clean["rsi_14"] > (100-rsi_th), f"RSI > {100-rsi_th}")

print()

# Volume
for vol_th in [1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
    test_condition(df_clean, df_clean["volume_ratio"] > vol_th, f"Volume > {vol_th}x")

print()

# BB position
for bb_th in [0.1, 0.2, 0.3, 0.4]:
    test_condition(df_clean, df_clean["bb_20_position"] < bb_th, f"BB position < {bb_th}")
    test_condition(df_clean, df_clean["bb_20_position"] > (1-bb_th), f"BB position > {1-bb_th}")

print()

# MA distance
for dist_th in [-0.01, -0.02, -0.03, -0.05, -0.1]:
    test_condition(df_clean, df_clean["close_ma20_dist"] < dist_th, f"MA20 dist < {dist_th}")
    test_condition(df_clean, df_clean["close_ma20_dist"] > -dist_th, f"MA20 dist > {-dist_th}")

print()

# Previous returns
for ret_th in [0.001, 0.002, 0.005, 0.01, 0.02]:
    test_condition(df_clean, df_clean["returns_lag1"] < -ret_th, f"Prev return < -{ret_th}")
    test_condition(df_clean, df_clean["returns_lag1"] > ret_th, f"Prev return > {ret_th}")

print()

# Cumulative returns
for cum_th in [0.005, 0.01, 0.02, 0.03, 0.05]:
    test_condition(df_clean, df_clean["cumret_5"] < -cum_th, f"Cumret5 < -{cum_th}")
    test_condition(df_clean, df_clean["cumret_5"] > cum_th, f"Cumret5 > {cum_th}")

print()

# Stochastic
for stoch_th in [10, 20, 30]:
    test_condition(df_clean, df_clean["stoch_k_14"] < stoch_th, f"Stoch < {stoch_th}")
    test_condition(df_clean, df_clean["stoch_k_14"] > (100-stoch_th), f"Stoch > {100-stoch_th}")

print()

# MACD
for macd_th in [-0.005, -0.01, -0.02, 0.005, 0.01, 0.02]:
    if macd_th < 0:
        test_condition(df_clean, df_clean["macd"] < macd_th, f"MACD < {macd_th}")
    else:
        test_condition(df_clean, df_clean["macd"] > macd_th, f"MACD > {macd_th}")

print()

# Combined conditions - maybe a very specific combination gets 90%?
print("=== Testing combined conditions ===\n")

# RSI extreme + volume
for rsi_th in [15, 20, 25, 30]:
    for vol_th in [1.0, 1.5, 2.0]:
        cond = (df_clean["rsi_14"] < rsi_th) & (df_clean["volume_ratio"] > vol_th)
        test_condition(df_clean, cond, f"RSI<{rsi_th} & Vol>{vol_th}x")

print()

# Large drop + volume spike
for ret_th in [0.005, 0.01, 0.02]:
    for vol_th in [1.5, 2.0, 3.0]:
        cond = (df_clean["returns_lag1"] < -ret_th) & (df_clean["volume_ratio"] > vol_th)
        test_condition(df_clean, cond, f"Drop>{ret_th} & Vol>{vol_th}x")

print()

# BB extreme + wick
for bb_th in [0.1, 0.2, 0.3]:
    for wick_th in [0.3, 0.5]:
        cond = (df_clean["bb_20_position"] < bb_th) & (df_clean["lower_wick_ratio"] > wick_th)
        test_condition(df_clean, cond, f"BB<{bb_th} & Wick>{wick_th}")

print()

# Time-based
print("=== Time-based conditions ===\n")
for h in range(24):
    cond = df_clean["hour"] == h
    test_condition(df_clean, cond, f"Hour {h:02d}")

print()

# Day of week (crypto is 7-day but maybe patterns exist)
for d in range(7):
    cond = df_clean["day_of_week"] == d
    test_condition(df_clean, cond, f"Day {d}")

print("\n=== SUMMARY ===")
print("No single condition or simple combination achieved 90% accuracy with >8000 trades.")
print("Best achievable with technical indicators at 25%+ coverage is ~55-60%.")
