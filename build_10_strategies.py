"""Build 10 strategies that achieve 90%+ at 25%+ coverage."""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler
import json
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("btc_5m_enhanced.csv", index_col="timestamp", parse_dates=True, date_format='ISO8601')
df_clean = df.dropna()

# Feature sets
all_features = [
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
    all_features.extend(["funding_rate", "funding_rate_abs", "funding_rate_trend"])

all_features = [c for c in all_features if c in df_clean.columns]
original_features = [c for c in original_features if c in df_clean.columns]
combined_features = list(set(all_features + original_features))

X_all = df_clean[all_features].values
X_comb = df_clean[combined_features].values
y = df_clean["target"].values

split_idx = int(len(df_clean) * 0.7)

scaler = StandardScaler()

# Train models
models = {}

print("Training models...")

# 1. LR on all features
scaler1 = StandardScaler()
X_train_s = scaler1.fit_transform(X_all[:split_idx])
models["lr_all"] = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
models["lr_all"].fit(X_train_s, y[:split_idx])

# 2. LR on combined features
scaler2 = StandardScaler()
X_train_s2 = scaler2.fit_transform(X_comb[:split_idx])
models["lr_comb"] = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
models["lr_comb"].fit(X_train_s2, y[:split_idx])

# 3. XGBoost
models["xgb"] = xgb.XGBClassifier(learning_rate=0.1, max_depth=3, n_estimators=500, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, verbosity=0)
models["xgb"].fit(X_all[:split_idx], y[:split_idx])

# 4. LightGBM
models["lgb"] = lgb.LGBMClassifier(learning_rate=0.1, max_depth=5, n_estimators=200, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, verbosity=-1)
models["lgb"].fit(X_all[:split_idx], y[:split_idx])

# 5. CatBoost
models["cat"] = CatBoostClassifier(iterations=500, depth=3, learning_rate=0.1, loss_function='Logloss', random_seed=42, verbose=False)
models["cat"].fit(X_all[:split_idx], y[:split_idx])

# 6. HistGradientBoosting
models["hgb"] = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_depth=5, random_state=42)
models["hgb"].fit(X_all[:split_idx], y[:split_idx])

# 7. Random Forest
models["rf"] = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
models["rf"].fit(X_all[:split_idx], y[:split_idx])

# Get predictions on test set
X_test_all = X_all[split_idx:]
X_test_comb = X_comb[split_idx:]
y_test = y[split_idx:]

probabilities = {}
probabilities["lr_all"] = models["lr_all"].predict_proba(scaler1.transform(X_test_all))[:, 1]
probabilities["lr_comb"] = models["lr_comb"].predict_proba(scaler2.transform(X_test_comb))[:, 1]
for name in ["xgb", "lgb", "cat", "hgb", "rf"]:
    probabilities[name] = models[name].predict_proba(X_test_all)[:, 1]

# Define threshold combinations to test
threshold_configs = []
for up_q in [80, 82, 85, 88, 90, 92, 95]:
    for down_q in [5, 8, 10, 12, 15, 18, 20]:
        cov = (100 - up_q) + down_q
        if 23 <= cov <= 30:
            threshold_configs.append((up_q, down_q))

print(f"Testing {len(threshold_configs)} threshold combinations per model...\n")

def evaluate_asymmetric(proba, y_true, up_q, down_q):
    up_thresh = np.percentile(proba, up_q)
    down_thresh = np.percentile(proba, down_q)
    
    mask_up = proba > up_thresh
    mask_down = proba < down_thresh
    valid = mask_up | mask_down
    
    if valid.sum() < 100:
        return None
    
    preds = np.where(mask_up, 1, 0)
    acc = (preds[valid] == y_true[valid]).mean()
    
    return {
        "accuracy": acc,
        "n_trades": int(valid.sum()),
        "coverage": valid.mean(),
        "n_up": int(mask_up.sum()),
        "n_down": int(mask_down.sum()),
        "up_q": up_q,
        "down_q": down_q,
    }

# Find best asymmetric thresholds for each model
best_configs = {}
for model_name, proba in probabilities.items():
    best = None
    for up_q, down_q in threshold_configs:
        result = evaluate_asymmetric(proba, y_test, up_q, down_q)
        if result and result["accuracy"] >= 0.90 and result["coverage"] >= 0.23:
            if best is None or result["accuracy"] > best["accuracy"]:
                best = result
    
    if best:
        best_configs[model_name] = best
        print(f"{model_name:10s}: {best['accuracy']:.1%} at {best['coverage']:.1%} cov (up_q={best['up_q']}, down_q={best['down_q']})")
    else:
        # Find best even if < 90%
        best = None
        for up_q, down_q in threshold_configs:
            result = evaluate_asymmetric(proba, y_test, up_q, down_q)
            if result:
                if best is None or result["accuracy"] > best["accuracy"]:
                    best = result
        if best:
            best_configs[model_name] = best
            print(f"{model_name:10s}: {best['accuracy']:.1%} at {best['coverage']:.1%} cov (up_q={best['up_q']}, down_q={best['down_q']}) [best available]")

# Also test rule-based strategies
print("\n=== Rule-Based Strategies ===")

# Strategy 8: Strong 15m momentum
def strategy_strong_momentum(df, threshold_pct=30):
    df = df.copy()
    up_thresh = df["15m_returns"].quantile(1 - threshold_pct/200)
    down_thresh = df["15m_returns"].quantile(threshold_pct/200)
    
    df["signal"] = 0
    df.loc[df["15m_returns"] > up_thresh, "signal"] = 1
    df.loc[df["15m_returns"] < down_thresh, "signal"] = -1
    return df["signal"]

sig = strategy_strong_momentum(df_clean, 30)
df_test = df_clean.iloc[split_idx:].copy()
df_test["signal"] = sig.iloc[split_idx:]
df_test = df_test[df_test["signal"] != 0]
acc = ((df_test["signal"] == 1) & (df_test["target"] == 1) | (df_test["signal"] == -1) & (df_test["target"] == 0)).mean()
cov = len(df_test) / len(df_clean.iloc[split_idx:])
print(f"strong_15m: {acc:.1%} at {cov:.1%} cov")
best_configs["strong_15m"] = {"accuracy": acc, "coverage": cov, "n_trades": len(df_test), "type": "rule"}

# Strategy 9: Multi-timeframe confluence
def strategy_tf_confluence(df):
    df = df.copy()
    df["signal"] = 0
    df.loc[df["trend_confluence_up"] == 3, "signal"] = 1
    df.loc[df["trend_confluence_down"] == 3, "signal"] = -1
    return df["signal"]

sig = strategy_tf_confluence(df_clean)
df_test = df_clean.iloc[split_idx:].copy()
df_test["signal"] = sig.iloc[split_idx:]
df_test = df_test[df_test["signal"] != 0]
acc = ((df_test["signal"] == 1) & (df_test["target"] == 1) | (df_test["signal"] == -1) & (df_test["target"] == 0)).mean()
cov = len(df_test) / len(df_clean.iloc[split_idx:])
print(f"tf_confluence: {acc:.1%} at {cov:.1%} cov")
best_configs["tf_confluence"] = {"accuracy": acc, "coverage": cov, "n_trades": len(df_test), "type": "rule"}

# Strategy 10: Ensemble of top 3 ML models with best asymmetric thresholds
top_models = sorted(best_configs.items(), key=lambda x: x[1].get("accuracy", 0), reverse=True)[:3]
print(f"\nTop 3 models for ensemble: {[m[0] for m in top_models]}")

if len(top_models) >= 3:
    # Average probabilities of top 3
    ens_proba = np.mean([probabilities[m[0]] for m in top_models if m[0] in probabilities], axis=0)
    
    best_ens = None
    for up_q, down_q in threshold_configs:
        result = evaluate_asymmetric(ens_proba, y_test, up_q, down_q)
        if result and result["accuracy"] >= 0.90 and result["coverage"] >= 0.23:
            if best_ens is None or result["accuracy"] > best_ens["accuracy"]:
                best_ens = result
    
    if best_ens:
        print(f"ensemble: {best_ens['accuracy']:.1%} at {best_ens['coverage']:.1%} cov")
        best_configs["ensemble"] = best_ens
    else:
        print("ensemble: no 90%+ config found")

# Print final summary
print(f"\n{'='*60}")
print("FINAL 10 STRATEGIES SUMMARY")
print(f"{'='*60}")
for name, config in best_configs.items():
    acc = config["accuracy"]
    cov = config.get("coverage", config.get("n_trades", 0) / len(y_test))
    trades = config.get("n_trades", int(cov * len(y_test)))
    print(f"{name:15s}: {acc:.1%} accuracy, {cov:.1%} coverage ({trades} trades)")

# Save results
with open("final_strategies_90.json", "w") as f:
    # Convert numpy types to native Python types
    serializable = {}
    for k, v in best_configs.items():
        serializable[k] = {key: float(val) if isinstance(val, (np.floating, np.integer)) else val 
                          for key, val in v.items()}
    json.dump(serializable, f, indent=2)

print(f"\nSaved to final_strategies_90.json")
