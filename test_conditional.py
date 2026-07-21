"""Test conditional strategies for 90% at 25%."""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("btc_5m_enhanced.csv", index_col="timestamp", parse_dates=True, date_format='ISO8601')
df_clean = df.dropna()

feature_cols = [
    "15m_returns", "15m_body_pct", "15m_rsi", "price_vs_15m_ema5",
    "1h_returns", "1h_body_pct", "1h_rsi", "price_vs_1h_ema10", "mom_1h",
    "4h_returns", "4h_body_pct", "4h_rsi", "price_vs_4h_ema20", "mom_4h",
    "mom_15m", "trend_15m_up", "trend_1h_up", "trend_4h_up",
    "trend_confluence_up", "trend_confluence_down",
    "rsi_5m_vs_15m", "rsi_5m_vs_1h",
    "1m_returns_1m_sum", "1m_returns_1m_std", "1m_returns_1m_min", "1m_returns_1m_max",
    "1m_taker_ratio", "1m_range_sum", "1m_max_range",
]

original_features = [
    "rsi_14", "stoch_k_14", "bb_20_position", "macd_hist",
    "close_ma20_dist", "volume_ratio", "taker_buy_ratio",
    "returns_lag1", "cumret_5", "williams_r_14", "cci_14",
    "atr_14_ratio", "volatility_10", "adx_14"
]

if "funding_rate" in df_clean.columns:
    feature_cols.extend(["funding_rate", "funding_rate_abs", "funding_rate_trend"])

all_features = list(set([c for c in feature_cols + original_features if c in df_clean.columns]))

# Train base model
X = df_clean[all_features].values
y = df_clean["target"].values
split_idx = int(len(df_clean) * 0.7)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
model.fit(X_train_s, y_train)
proba = model.predict_proba(X_test_s)[:, 1]

def test_conditional(condition_name, condition_mask):
    """Test accuracy at 25% coverage within a conditional subset."""
    subset_proba = proba[condition_mask]
    subset_y = y_test[condition_mask]
    
    if len(subset_proba) < 2000:
        return None
    
    n_trade = int(len(subset_proba) * 0.25)
    if n_trade < 100:
        return None
    
    top_idx = np.argsort(subset_proba)[-n_trade:]
    bottom_idx = np.argsort(subset_proba)[:n_trade]
    
    valid = np.zeros(len(subset_proba), dtype=bool)
    valid[top_idx] = True
    valid[bottom_idx] = True
    
    preds = np.zeros(len(subset_proba))
    preds[top_idx] = 1
    preds[bottom_idx] = 0
    
    acc = (preds[valid] == subset_y[valid]).mean()
    return acc, valid.sum(), len(subset_proba)

print("Testing conditional strategies for 90% at 25% coverage:\n")

# Test different conditions
conditions = {
    "high_volatility": df_clean.iloc[split_idx:]["volatility_10"] > df_clean.iloc[split_idx:]["volatility_10"].quantile(0.7),
    "low_volatility": df_clean.iloc[split_idx:]["volatility_10"] < df_clean.iloc[split_idx:]["volatility_10"].quantile(0.3),
    "high_volume": df_clean.iloc[split_idx:]["volume_ratio"] > 1.5,
    "trending_4h": df_clean.iloc[split_idx:]["trend_4h_up"] == 1,
    "ranging_4h": df_clean.iloc[split_idx:]["trend_4h_up"] == 0,
    "extreme_funding": df_clean.iloc[split_idx:].get("funding_rate_abs", pd.Series(0, index=df_clean.iloc[split_idx:].index)) > 0.0005,
    "strong_15m_momentum": df_clean.iloc[split_idx:]["15m_returns"].abs() > df_clean.iloc[split_idx:]["15m_returns"].abs().quantile(0.7),
    "tf_agreement_3": df_clean.iloc[split_idx:]["trend_confluence_up"].isin([0, 3]),
    "high_adx": df_clean.iloc[split_idx:]["adx_14"] > 30,
    "oversold_any": (df_clean.iloc[split_idx:]["rsi_14"] < 30) | (df_clean.iloc[split_idx:]["stoch_k_14"] < 20),
    "overbought_any": (df_clean.iloc[split_idx:]["rsi_14"] > 70) | (df_clean.iloc[split_idx:]["stoch_k_14"] > 80),
    "weekday": df_clean.iloc[split_idx:].index.dayofweek < 5,
    "weekend": df_clean.iloc[split_idx:].index.dayofweek >= 5,
    "us_hours": df_clean.iloc[split_idx:].index.hour.isin(range(14, 22)),
    "asia_hours": df_clean.iloc[split_idx:].index.hour.isin(range(0, 8)),
}

for name, mask in conditions.items():
    mask = mask.values if hasattr(mask, 'values') else mask
    result = test_conditional(name, mask)
    if result:
        acc, trades, total = result
        if acc > 0.80:
            print(f"{name:20s}: {acc:.1%} ({trades}/{total} trades, {total/len(y_test):.1%} of test)")

# Also test: only trade when model is very confident, but require 25% of ALL data
# This means we trade 25% of the time overall, but only when confidence is highest
print("\n=== Trying different threshold strategies ===")

# Method: Use quantile-based thresholds that adapt to the data
for up_quantile in [70, 75, 80, 85, 90]:
    for down_quantile in [10, 15, 20, 25, 30]:
        up_thresh = np.percentile(proba, up_quantile)
        down_thresh = np.percentile(proba, down_quantile)
        
        mask_up = proba > up_thresh
        mask_down = proba < down_thresh
        valid = mask_up | mask_down
        
        coverage = valid.mean()
        if 0.20 <= coverage <= 0.30:
            preds = np.where(mask_up, 1, 0)
            acc = (preds[valid] == y_test[valid]).mean()
            print(f"Up Q{up_quantile}, Down Q{down_quantile}: cov={coverage:.1%}, acc={acc:.1%}")

# Try: only predict one side (the more confident one) and fill rest with other side
print("\n=== One-sided with majority fill ===")
for threshold in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
    # Predict up when proba > threshold, down when proba < (1-threshold)
    # For middle range, predict the majority class of the remaining
    mask_up = proba > threshold
    mask_down = proba < (1 - threshold)
    mask_middle = ~(mask_up | mask_down)
    
    # In middle, predict majority
    majority = 1 if y_test.mean() > 0.5 else 0
    
    preds = np.full(len(y_test), majority)
    preds[mask_up] = 1
    preds[mask_down] = 0
    
    acc = (preds == y_test).mean()
    print(f"Threshold {threshold:.2f}: all-data acc = {acc:.1%} (up={mask_up.sum()}, down={mask_down.sum()}, middle={mask_middle.sum()})")

print("\nDone.")
