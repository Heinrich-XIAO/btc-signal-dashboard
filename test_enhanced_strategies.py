"""Test strategies using multi-timeframe + funding rate features."""
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("btc_5m_enhanced.csv", index_col="timestamp", parse_dates=True, date_format='ISO8601')
df_clean = df.dropna()

print(f"Dataset: {len(df_clean)} rows, {len(df_clean.columns)} columns")
print(f"Baseline: {df_clean['target'].mean():.3f}")
print()

# Test 1: Higher timeframe momentum continuation
print("=== 1. Higher Timeframe Momentum ===")

# 15m return continuation
for threshold_pct in [10, 20, 25, 30, 40]:
    up_threshold = df_clean["15m_returns"].quantile(1 - threshold_pct/100)
    down_threshold = df_clean["15m_returns"].quantile(threshold_pct/100)
    
    mask_up = df_clean["15m_returns"] > up_threshold
    mask_down = df_clean["15m_returns"] < down_threshold
    valid = mask_up | mask_down
    
    if valid.sum() > 0:
        preds = np.where(mask_up, 1, 0)
        acc = (preds[valid] == df_clean.loc[valid, "target"]).mean()
        print(f"  15m momentum top/bottom {threshold_pct}%: {acc:.3f} accuracy ({valid.sum()} trades)")

print()

# Test 2: Multi-timeframe confluence
print("=== 2. Multi-Timeframe Confluence ===")

# When 15m, 1h, and 4h all agree
df_clean["tf_agreement"] = np.sign(df_clean["15m_returns"]) + np.sign(df_clean["1h_returns"]) + np.sign(df_clean["4h_returns"])

for agreement in [3, 2, 1, -1, -2, -3]:
    if agreement > 0:
        mask = df_clean["tf_agreement"] >= agreement
        pred = 1
    else:
        mask = df_clean["tf_agreement"] <= agreement
        pred = 0
    
    if mask.sum() > 100:
        acc = (pred == df_clean.loc[mask, "target"]).mean()
        print(f"  TF agreement {agreement:2d}: {acc:.3f} accuracy ({mask.sum()} trades)")

print()

# Test 3: Funding rate + trend
print("=== 3. Funding Rate + Trend ===")

if "funding_rate" in df_clean.columns:
    # Extreme funding + trend continuation
    for fund_th in [0.0001, 0.0005, 0.001, 0.002]:
        for trend_col in ["15m_returns", "1h_returns"]:
            # High funding + positive trend = continue up
            mask = (df_clean["funding_rate"].abs() > fund_th) & (df_clean[trend_col] > 0)
            if mask.sum() > 100:
                acc = df_clean.loc[mask, "target"].mean()
                print(f"  High funding + {trend_col}>0: {acc:.3f} ({mask.sum()} trades)")

print()

# Test 4: ML on multi-timeframe features only
print("=== 4. ML on Multi-Timeframe Features ===")

# Select the best new features
new_features = [
    "15m_returns", "15m_body_pct", "15m_rsi", "15m_ema5", "15m_ema10", "15m_ema20",
    "1h_returns", "1h_body_pct", "1h_rsi", "1h_ema5", "1h_ema10", "1h_ema20",
    "4h_returns", "4h_body_pct", "4h_rsi", "4h_ema5", "4h_ema10", "4h_ema20",
    "price_vs_15m_ema5", "price_vs_1h_ema10", "price_vs_4h_ema20",
    "mom_15m", "mom_1h", "mom_4h",
    "trend_15m_up", "trend_1h_up", "trend_4h_up",
    "trend_confluence_up", "trend_confluence_down",
    "rsi_5m_vs_15m", "rsi_5m_vs_1h", "rsi_15m_vs_1h",
    "rsi_all_oversold", "rsi_all_overbought",
]

# Add funding features if available
if "funding_rate" in df_clean.columns:
    new_features.extend(["funding_rate", "funding_rate_abs", "funding_rate_trend", "funding_extreme"])

# Add 1m microstructure
micro_features = [c for c in df_clean.columns if c.startswith("1m_")]
new_features.extend(micro_features[:10])  # Limit to avoid too many

# Filter to existing columns
feature_cols = [c for c in new_features if c in df_clean.columns]
print(f"Using {len(feature_cols)} features: {feature_cols[:10]}...")

X = df_clean[feature_cols].values
y = df_clean["target"].values

split_idx = int(len(df_clean) * 0.7)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# Logistic Regression
lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
lr.fit(X_train_s, y_train)
proba_lr = lr.predict_proba(X_test_s)[:, 1]

# Random Forest
rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
proba_rf = rf.predict_proba(X_test)[:, 1]

# HistGradientBoosting
hgb = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_depth=5, random_state=42)
hgb.fit(X_train, y_train)
proba_hgb = hgb.predict_proba(X_test)[:, 1]

print()
print("Coverage vs Accuracy:")
print(f"{'Model':12s} {'Cov':>5s} {'Trades':>7s} {'Acc':>6s}")
print("-" * 35)

for model_name, proba in [("LR", proba_lr), ("RF", proba_rf), ("HGB", proba_hgb)]:
    for cov in [0.10, 0.15, 0.20, 0.25, 0.30]:
        n_trade = int(len(X_test) * cov)
        # Take top N and bottom N
        top_idx = np.argsort(proba)[-n_trade:]
        bottom_idx = np.argsort(proba)[:n_trade]
        
        valid = np.zeros(len(y_test), dtype=bool)
        valid[top_idx] = True
        valid[bottom_idx] = True
        
        preds = np.zeros(len(y_test))
        preds[top_idx] = 1
        preds[bottom_idx] = 0
        
        acc = (preds[valid] == y_test[valid]).mean()
        print(f"{model_name:12s} {cov:>5.0%} {valid.sum():>7d} {acc:>6.1%}")
    print()

# Feature importance from RF
importances = rf.feature_importances_
feature_importance = sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)
print("Top 15 feature importances (Random Forest):")
for feat, imp in feature_importance[:15]:
    print(f"  {feat:25s}: {imp:.4f}")

print("\nDone.")
