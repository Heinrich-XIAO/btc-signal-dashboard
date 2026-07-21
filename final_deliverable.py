"""Final deliverable: 10 strategies at 25% coverage + ultimate strategy."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from strategies import STRATEGIES, evaluate_signal
from extreme_strategies import EXTREME_STRATEGIES

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (16, 10)
plt.rcParams["font.size"] = 11

df = pd.read_csv("btc_5m_features.csv", index_col="timestamp", parse_dates=True)

print("=" * 70)
print("FINAL DELIVERABLE: 10 STRATEGIES WITH 25%+ COVERAGE")
print("=" * 70)

# We need strategies with >= 25% coverage.
# With 35,088 candles, that's ~8,772 trades.
# Best achievable accuracy at this coverage: ~55-60%

# Strategy 1: RSI Momentum (trade with RSI direction, not reversal)
def strategy_rsi_momentum(df, rsi_period=14, oversold=40, overbought=60):
    df = df.copy()
    rsi_col = f"rsi_{rsi_period}"
    df["signal"] = 0
    df.loc[df[rsi_col] < oversold, "signal"] = -1  # Weak momentum = down
    df.loc[df[rsi_col] > overbought, "signal"] = 1  # Strong momentum = up
    return df["signal"]

# Strategy 2: Bollinger Band Position (trade toward middle)
def strategy_bb_middle(df, bb_period=20, low=0.3, high=0.7):
    df = df.copy()
    pos_col = f"bb_{bb_period}_position"
    df["signal"] = 0
    df.loc[df[pos_col] < low, "signal"] = 1   # Below low = mean reversion up
    df.loc[df[pos_col] > high, "signal"] = -1 # Above high = mean reversion down
    return df["signal"]

# Strategy 3: Stochastic Momentum
def strategy_stoch_momentum(df, stoch_period=14, low=30, high=70):
    df = df.copy()
    k_col = f"stoch_k_{stoch_period}"
    df["signal"] = 0
    df.loc[df[k_col] < low, "signal"] = -1
    df.loc[df[k_col] > high, "signal"] = 1
    return df["signal"]

# Strategy 4: Volume Spike with Direction
def strategy_volume_direction(df, vol_threshold=1.0):
    df = df.copy()
    df["signal"] = 0
    df.loc[(df["volume_ratio"] > vol_threshold) & (df["returns"] > 0), "signal"] = 1
    df.loc[(df["volume_ratio"] > vol_threshold) & (df["returns"] < 0), "signal"] = -1
    return df["signal"]

# Strategy 5: MA Distance (simplified)
def strategy_ma_simple(df, ma_period=20, low=-0.01, high=0.01):
    df = df.copy()
    dist_col = f"close_ma{ma_period}_dist"
    df["signal"] = 0
    df.loc[df[dist_col] < low, "signal"] = 1
    df.loc[df[dist_col] > high, "signal"] = -1
    return df["signal"]

# Strategy 6: MACD Histogram
def strategy_macd_hist(df, hist_threshold=0):
    df = df.copy()
    df["signal"] = 0
    df.loc[df["macd_hist"] > hist_threshold, "signal"] = 1
    df.loc[df["macd_hist"] < -hist_threshold, "signal"] = -1
    return df["signal"]

# Strategy 7: Williams R
def strategy_williams(df, period=14, low=-80, high=-20):
    df = df.copy()
    r_col = f"williams_r_{period}"
    df["signal"] = 0
    df.loc[df[r_col] < low, "signal"] = 1
    df.loc[df[r_col] > high, "signal"] = -1
    return df["signal"]

# Strategy 8: CCI
def strategy_cci(df, period=14, low=-100, high=100):
    df = df.copy()
    cci_col = f"cci_{period}"
    df["signal"] = 0
    df.loc[df[cci_col] < low, "signal"] = 1
    df.loc[df[cci_col] > high, "signal"] = -1
    return df["signal"]

# Strategy 9: Previous Return Continuation
def strategy_prev_continuation(df, lag=1, threshold=0.001):
    df = df.copy()
    ret_col = f"returns_lag{lag}"
    df["signal"] = 0
    df.loc[df[ret_col] > threshold, "signal"] = 1
    df.loc[df[ret_col] < -threshold, "signal"] = -1
    return df["signal"]

# Strategy 10: Cumulative Return Momentum
def strategy_cumret_momentum(df, period=5, threshold=0.002):
    df = df.copy()
    cum_col = f"cumret_{period}"
    df["signal"] = 0
    df.loc[df[cum_col] > threshold, "signal"] = 1
    df.loc[df[cum_col] < -threshold, "signal"] = -1
    return df["signal"]

TEN_STRATEGIES = {
    "rsi_momentum": strategy_rsi_momentum,
    "bb_mean_reversion": strategy_bb_middle,
    "stoch_momentum": strategy_stoch_momentum,
    "volume_direction": strategy_volume_direction,
    "ma_distance": strategy_ma_simple,
    "macd_momentum": strategy_macd_hist,
    "williams_r": strategy_williams,
    "cci": strategy_cci,
    "prev_continuation": strategy_prev_continuation,
    "cumret_momentum": strategy_cumret_momentum,
}

# Test and optimize each for 25% coverage
min_coverage = 0.25
min_trades = int(len(df.dropna()) * min_coverage)
print(f"Minimum trades required for 25% coverage: {min_trades}")
print()

final_results = []

# RSI Momentum - tune thresholds to get ~25% coverage
for oversold in [30, 35, 40, 45, 50]:
    overbought = 100 - oversold
    signal = strategy_rsi_momentum(df, oversold=oversold, overbought=overbought)
    df_test = df.copy()
    df_test["signal"] = signal
    result = evaluate_signal(df_test, "signal", min_trades=min_trades)
    if result["n_trades"] >= min_trades:
        final_results.append({"name": "rsi_momentum", "params": {"oversold": oversold, "overbought": overbought}, **result})
        break

# BB
for low in [0.2, 0.25, 0.3, 0.35, 0.4]:
    high = 1 - low
    signal = strategy_bb_middle(df, low=low, high=high)
    df_test = df.copy()
    df_test["signal"] = signal
    result = evaluate_signal(df_test, "signal", min_trades=min_trades)
    if result["n_trades"] >= min_trades:
        final_results.append({"name": "bb_mean_reversion", "params": {"low": low, "high": high}, **result})
        break

# Stoch
for low in [20, 25, 30, 35, 40]:
    high = 100 - low
    signal = strategy_stoch_momentum(df, low=low, high=high)
    df_test = df.copy()
    df_test["signal"] = signal
    result = evaluate_signal(df_test, "signal", min_trades=min_trades)
    if result["n_trades"] >= min_trades:
        final_results.append({"name": "stoch_momentum", "params": {"low": low, "high": high}, **result})
        break

# Volume
for vol in [0.8, 1.0, 1.2, 1.5]:
    signal = strategy_volume_direction(df, vol_threshold=vol)
    df_test = df.copy()
    df_test["signal"] = signal
    result = evaluate_signal(df_test, "signal", min_trades=min_trades)
    if result["n_trades"] >= min_trades:
        final_results.append({"name": "volume_direction", "params": {"vol_threshold": vol}, **result})
        break

# MA
for dist in [0.005, 0.01, 0.015, 0.02, 0.03]:
    signal = strategy_ma_simple(df, low=-dist, high=dist)
    df_test = df.copy()
    df_test["signal"] = signal
    result = evaluate_signal(df_test, "signal", min_trades=min_trades)
    if result["n_trades"] >= min_trades:
        final_results.append({"name": "ma_distance", "params": {"dist": dist}, **result})
        break

# MACD
for threshold in [0, 0.0001, 0.0005, 0.001, 0.002]:
    signal = strategy_macd_hist(df, hist_threshold=threshold)
    df_test = df.copy()
    df_test["signal"] = signal
    result = evaluate_signal(df_test, "signal", min_trades=min_trades)
    if result["n_trades"] >= min_trades:
        final_results.append({"name": "macd_momentum", "params": {"threshold": threshold}, **result})
        break

# Williams
for level in [70, 75, 80, 85, 90]:
    signal = strategy_williams(df, low=-level, high=-(100-level))
    df_test = df.copy()
    df_test["signal"] = signal
    result = evaluate_signal(df_test, "signal", min_trades=min_trades)
    if result["n_trades"] >= min_trades:
        final_results.append({"name": "williams_r", "params": {"level": level}, **result})
        break

# CCI
for level in [80, 100, 120, 150]:
    signal = strategy_cci(df, low=-level, high=level)
    df_test = df.copy()
    df_test["signal"] = signal
    result = evaluate_signal(df_test, "signal", min_trades=min_trades)
    if result["n_trades"] >= min_trades:
        final_results.append({"name": "cci", "params": {"level": level}, **result})
        break

# Prev continuation
for threshold in [0.000, 0.0005, 0.001, 0.002, 0.003]:
    signal = strategy_prev_continuation(df, threshold=threshold)
    df_test = df.copy()
    df_test["signal"] = signal
    result = evaluate_signal(df_test, "signal", min_trades=min_trades)
    if result["n_trades"] >= min_trades:
        final_results.append({"name": "prev_continuation", "params": {"threshold": threshold}, **result})
        break

# Cumret
for threshold in [0.000, 0.001, 0.002, 0.003, 0.005]:
    signal = strategy_cumret_momentum(df, threshold=threshold)
    df_test = df.copy()
    df_test["signal"] = signal
    result = evaluate_signal(df_test, "signal", min_trades=min_trades)
    if result["n_trades"] >= min_trades:
        final_results.append({"name": "cumret_momentum", "params": {"threshold": threshold}, **result})
        break

# Print results
print(f"{'='*70}")
print("RESULTS FOR 10 STRATEGIES (25%+ COVERAGE)")
print(f"{'='*70}")
for r in final_results:
    print(f"{r['name']:20s}: {r['accuracy']*100:.1f}% ({r['n_trades']} trades)")

# Also show the rare high-accuracy strategies
print()
print(f"{'='*70}")
print("RARE HIGH-ACCURACY STRATEGIES (for comparison)")
print(f"{'='*70}")

with open("optimization_round_2.json", "r") as f:
    rare = json.load(f)

for name, res in rare.items():
    if res and res[0]["accuracy"] >= 0.8 and name != "ensembles":
        print(f"{name:30s}: {res[0]['accuracy']*100:.1f}% ({res[0]['n_trades']} trades)")

# Save results
with open("final_10_strategies.json", "w") as f:
    json.dump(final_results, f, indent=2, default=str)

print()
print("Results saved to final_10_strategies.json")
