"""Engineer advanced features from multi-timeframe + funding rate data."""
import pandas as pd
import numpy as np

# Load data - use features.csv as base since it has all technical features
df_5m = pd.read_csv("btc_5m_features.csv", index_col="timestamp", parse_dates=True, date_format='ISO8601')
df_1m = pd.read_csv("btc_1m_data.csv", index_col="timestamp", parse_dates=True, date_format='ISO8601')
df_15m = pd.read_csv("btc_15m_data.csv", index_col="timestamp", parse_dates=True, date_format='ISO8601')
df_1h = pd.read_csv("btc_1h_data.csv", index_col="timestamp", parse_dates=True, date_format='ISO8601')
df_4h = pd.read_csv("btc_4h_data.csv", index_col="timestamp", parse_dates=True, date_format='ISO8601')
df_funding = pd.read_csv("btc_funding_rates.csv", index_col="timestamp", parse_dates=True, date_format='ISO8601')

print("Data loaded:")
print(f"  5m: {len(df_5m)}")
print(f"  1m: {len(df_1m)}")
print(f"  15m: {len(df_15m)}")
print(f"  1h: {len(df_1h)}")
print(f"  4h: {len(df_4h)}")
print(f"  Funding: {len(df_funding)}")

# Start with 5m data
df = df_5m.copy()
df["returns"] = df["close"].pct_change()
df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)

# === FUNDING RATE FEATURES ===
print("\nAdding funding rate features...")

# Ensure funding index is datetime
df_funding.index = pd.to_datetime(df_funding.index)

# Merge funding rates (forward fill to 5m intervals)
df_funding_resampled = df_funding.resample("5min").ffill()
df = df.join(df_funding_resampled, how="left")
df["fundingRate"] = df["fundingRate"].ffill()

# Funding features
if "fundingRate" in df.columns:
    df["funding_rate"] = df["fundingRate"]
    df["funding_rate_abs"] = df["funding_rate"].abs()
    df["funding_rate_sign"] = np.sign(df["funding_rate"])
    df["funding_rate_ma3"] = df["funding_rate"].rolling(3).mean()
    df["funding_rate_trend"] = df["funding_rate"].diff(3)
    df["funding_extreme"] = (df["funding_rate_abs"] > df["funding_rate_abs"].quantile(0.9)).astype(int)
    df["funding_positive"] = (df["funding_rate"] > 0).astype(int)

# === 1M MICROSTRUCTURE FEATURES ===
print("Adding 1m microstructure features...")

# Resample 1m to 5m and merge
df_1m["returns_1m"] = df_1m["close"].pct_change()
df_1m["range_1m"] = df_1m["high"] - df_1m["low"]
df_1m["volume_1m"] = df_1m["volume"]

# Aggregate 1m to 5m
agg_1m = df_1m.resample("5min").agg({
    "returns_1m": ["sum", "std", "min", "max", "count"],
    "range_1m": ["sum", "mean", "max"],
    "volume_1m": ["sum", "mean"],
    "taker_buy_volume": ["sum"],
})
agg_1m.columns = ["_".join(c).strip() for c in agg_1m.columns]
agg_1m = agg_1m.add_prefix("1m_")

df = df.join(agg_1m, how="left")

# More 1m features
if "1m_returns_1m_sum" in df.columns:
    df["1m_momentum"] = df["1m_returns_1m_sum"]
    df["1m_volatility"] = df["1m_returns_1m_std"]
    df["1m_range_sum"] = df["1m_range_1m_sum"]
    df["1m_max_range"] = df["1m_range_1m_max"]
    df["1m_candle_count"] = df["1m_returns_1m_count"]
    df["1m_taker_ratio"] = df["1m_taker_buy_volume_sum"] / (df["1m_volume_1m_sum"] + 1e-8)

# === MULTI-TIMEFRAME FEATURES ===
print("Adding multi-timeframe features...")

def add_tf_features(df, tf_data, prefix):
    """Add features from higher timeframe."""
    tf_data = tf_data.copy()
    tf_data["returns"] = tf_data["close"].pct_change()
    tf_data["range"] = tf_data["high"] - tf_data["low"]
    tf_data["body"] = tf_data["close"] - tf_data["open"]
    tf_data["body_pct"] = tf_data["body"] / tf_data["open"]
    
    # RSI
    delta = tf_data["close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    tf_data["rsi"] = 100 - (100 / (1 + rs))
    
    # EMAs
    for span in [5, 10, 20]:
        tf_data[f"ema{span}"] = tf_data["close"].ewm(span=span, adjust=False).mean()
    
    # Resample to 5m (forward fill)
    tf_resampled = tf_data[["close", "returns", "range", "body_pct", "rsi", "ema5", "ema10", "ema20"]].resample("5min").ffill()
    tf_resampled = tf_resampled.add_prefix(f"{prefix}_")
    
    return tf_resampled

df_15m_tf = add_tf_features(df, df_15m, "15m")
df_1h_tf = add_tf_features(df, df_1h, "1h")
df_4h_tf = add_tf_features(df, df_4h, "4h")

df = df.join(df_15m_tf, how="left")
df = df.join(df_1h_tf, how="left")
df = df.join(df_4h_tf, how="left")

# Multi-timeframe alignment features
if "15m_rsi" in df.columns and "1h_rsi" in df.columns:
    df["rsi_5m_vs_15m"] = df.get("rsi_14", pd.Series(np.nan, index=df.index)) - df["15m_rsi"]
    df["rsi_5m_vs_1h"] = df.get("rsi_14", pd.Series(np.nan, index=df.index)) - df["1h_rsi"]
    df["rsi_15m_vs_1h"] = df["15m_rsi"] - df["1h_rsi"]
    
    # Timeframe confluence: all RSI pointing same direction
    df["rsi_5m_oversold"] = (df.get("rsi_14", pd.Series(50, index=df.index)) < 30).astype(int)
    df["rsi_15m_oversold"] = (df["15m_rsi"] < 30).astype(int)
    df["rsi_1h_oversold"] = (df["1h_rsi"] < 30).astype(int)
    df["rsi_4h_oversold"] = (df["4h_rsi"] < 30).astype(int)
    
    df["rsi_all_oversold"] = df["rsi_5m_oversold"] + df["rsi_15m_oversold"] + df["rsi_1h_oversold"] + df["rsi_4h_oversold"]
    
    df["rsi_5m_overbought"] = (df.get("rsi_14", pd.Series(50, index=df.index)) > 70).astype(int)
    df["rsi_15m_overbought"] = (df["15m_rsi"] > 70).astype(int)
    df["rsi_1h_overbought"] = (df["1h_rsi"] > 70).astype(int)
    df["rsi_4h_overbought"] = (df["4h_rsi"] > 70).astype(int)
    
    df["rsi_all_overbought"] = df["rsi_5m_overbought"] + df["rsi_15m_overbought"] + df["rsi_1h_overbought"] + df["rsi_4h_overbought"]

# Higher timeframe trend direction
if "4h_ema20" in df.columns:
    df["trend_4h_up"] = (df["4h_close"] > df["4h_ema20"]).astype(int)
    df["trend_1h_up"] = (df["1h_close"] > df["1h_ema20"]).astype(int)
    df["trend_15m_up"] = (df["15m_close"] > df["15m_ema20"]).astype(int)
    
    df["trend_confluence_up"] = df["trend_4h_up"] + df["trend_1h_up"] + df["trend_15m_up"]
    df["trend_confluence_down"] = 3 - df["trend_confluence_up"]

# 5m price relative to higher timeframe EMAs
if "1h_ema10" in df.columns:
    df["price_vs_1h_ema10"] = df["close"] / df["1h_ema10"]
    df["price_vs_4h_ema20"] = df["close"] / df["4h_ema20"]
    df["price_vs_15m_ema5"] = df["close"] / df["15m_ema5"]

# Higher timeframe momentum
if "4h_returns" in df.columns:
    df["mom_4h"] = df["4h_returns"].rolling(5).sum()
    df["mom_1h"] = df["1h_returns"].rolling(5).sum()
    df["mom_15m"] = df["15m_returns"].rolling(5).sum()

# Higher timeframe volume features
if "15m_range" in df.columns:
    df["volatility_15m"] = df["15m_range"].rolling(10).mean() / df["15m_close"]
    df["volatility_1h"] = df["1h_range"].rolling(10).mean() / df["1h_close"]

# === CROSS-FEATURES ===
print("Adding cross-features...")

# Funding rate x RSI
if "funding_rate" in df.columns and "rsi_14" in df.columns:
    df["funding_x_rsi"] = df["funding_rate"] * df.get("rsi_14", 0)
    df["funding_x_price_change"] = df["funding_rate"] * df["returns"]

# Volume x funding
if "funding_rate" in df.columns:
    df["funding_x_volume"] = df["funding_rate"] * df["volume_ratio"]

# Microstructure x trend
if "1m_momentum" in df.columns and "trend_4h_up" in df.columns:
    df["micro_trend_align"] = np.where(
        df["trend_4h_up"] == 1,
        df["1m_momentum"],
        -df["1m_momentum"]
    )

# Save enhanced features
print(f"\nSaving enhanced dataset with {len(df.columns)} columns...")
df.to_csv("btc_5m_enhanced.csv")

print(f"Shape: {df.shape}")
print("Enhanced dataset saved to btc_5m_enhanced.csv")

# Quick preview of new feature correlations
print("\nTop new features by correlation with target:")
df_clean = df.dropna()
if len(df_clean) > 0:
    # Only numeric columns
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    corr = df_clean[numeric_cols].corr()["target"].drop("target")
    new_features = [c for c in corr.index if any(x in c for x in ["funding", "1m_", "15m_", "1h_", "4h_", "trend_", "rsi_all_", "mom_"])]
    if new_features:
        print(corr[new_features].dropna().sort_values(key=lambda x: np.abs(x), ascending=False).head(15))
    else:
        print("No new features found in correlation check")
