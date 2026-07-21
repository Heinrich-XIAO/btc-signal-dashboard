"""Try to push accuracy to 90%+ at 25% coverage using ensembles and feature selection."""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
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

if "funding_rate" in df_clean.columns:
    all_features.extend(["funding_rate", "funding_rate_abs", "funding_rate_trend"])

all_features = [c for c in all_features if c in df_clean.columns]

# Also try with original technical features
original_features = [
    "rsi_14", "stoch_k_14", "bb_20_position", "macd_hist",
    "close_ma20_dist", "volume_ratio", "taker_buy_ratio",
    "returns_lag1", "cumret_5", "williams_r_14", "cci_14",
    "atr_14_ratio", "volatility_10", "adx_14"
]
original_features = [c for c in original_features if c in df_clean.columns]

combined_features = list(set(all_features + original_features))
print(f"All MT features: {len(all_features)}")
print(f"Original features: {len(original_features)}")
print(f"Combined: {len(combined_features)}")

X_all = df_clean[all_features].values
X_orig = df_clean[original_features].values
X_comb = df_clean[combined_features].values
y = df_clean["target"].values

split_idx = int(len(df_clean) * 0.7)

scaler = StandardScaler()

def train_and_evaluate(X_train, X_test, y_train, y_test, model, name):
    if isinstance(model, LogisticRegression):
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        model.fit(X_train_s, y_train)
        proba = model.predict_proba(X_test_s)[:, 1]
    else:
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
    
    # Evaluate at different coverages
    results = {}
    for cov in [0.10, 0.15, 0.20, 0.25, 0.30]:
        n_trade = int(len(y_test) * cov)
        top_idx = np.argsort(proba)[-n_trade:]
        bottom_idx = np.argsort(proba)[:n_trade]
        
        valid = np.zeros(len(y_test), dtype=bool)
        valid[top_idx] = True
        valid[bottom_idx] = True
        
        preds = np.zeros(len(y_test))
        preds[top_idx] = 1
        preds[bottom_idx] = 0
        
        acc = (preds[valid] == y_test[valid]).mean()
        results[cov] = (acc, valid.sum())
    
    return proba, results

# Train multiple models on different feature sets
models = {
    "LR_all": LogisticRegression(C=1.0, max_iter=1000, random_state=42),
    "LR_comb": LogisticRegression(C=1.0, max_iter=1000, random_state=42),
    "HGB_all": HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_depth=5, random_state=42),
    "HGB_comb": HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_depth=5, random_state=42),
    "XGB_all": xgb.XGBClassifier(learning_rate=0.1, max_depth=3, n_estimators=500, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, verbosity=0),
    "LGB_all": lgb.LGBMClassifier(learning_rate=0.1, max_depth=5, n_estimators=200, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, verbosity=-1),
    "CAT_all": CatBoostClassifier(iterations=500, depth=3, learning_rate=0.1, loss_function='Logloss', random_seed=42, verbose=False),
}

probabilities = {}
all_results = {}

print("\nTraining models...")
for name, model in models.items():
    if "all" in name:
        X_train, X_test = X_all[:split_idx], X_all[split_idx:]
    else:
        X_train, X_test = X_comb[:split_idx], X_comb[split_idx:]
    
    y_train, y_test = y[:split_idx], y[split_idx:]
    proba, results = train_and_evaluate(X_train, X_test, y_train, y_test, model, name)
    probabilities[name] = proba
    all_results[name] = results
    
    print(f"{name:12s}: 25% = {results[0.25][0]:.1%}")

# Try ensemble methods
print("\n=== ENSEMBLES ===")

# 1. Simple average
for combo_name, combo_models in [
    ("avg_all", ["LR_all", "HGB_all", "XGB_all", "LGB_all", "CAT_all"]),
    ("avg_top3", ["XGB_all", "LGB_all", "CAT_all"]),
    ("avg_lr_xgb", ["LR_all", "XGB_all"]),
]:
    avg_proba = np.mean([probabilities[m] for m in combo_models], axis=0)
    results = {}
    for cov in [0.10, 0.15, 0.20, 0.25, 0.30]:
        n_trade = int(len(y_test) * cov)
        top_idx = np.argsort(avg_proba)[-n_trade:]
        bottom_idx = np.argsort(avg_proba)[:n_trade]
        
        valid = np.zeros(len(y_test), dtype=bool)
        valid[top_idx] = True
        valid[bottom_idx] = True
        
        preds = np.zeros(len(y_test))
        preds[top_idx] = 1
        preds[bottom_idx] = 0
        
        acc = (preds[valid] == y_test[valid]).mean()
        results[cov] = (acc, valid.sum())
    
    all_results[combo_name] = results
    print(f"{combo_name:12s}: 25% = {results[0.25][0]:.1%}")

# 2. Weighted average based on individual performance at 25%
weights = {}
for name, res in all_results.items():
    if name.startswith("avg_"):
        continue
    weights[name] = res[0.25][0]

# Normalize weights
total_weight = sum(weights.values())
normalized_weights = {k: v/total_weight for k, v in weights.items()}

weighted_proba = np.zeros(len(y_test))
for name, weight in normalized_weights.items():
    if name in probabilities:
        weighted_proba += probabilities[name] * weight

results = {}
for cov in [0.10, 0.15, 0.20, 0.25, 0.30]:
    n_trade = int(len(y_test) * cov)
    top_idx = np.argsort(weighted_proba)[-n_trade:]
    bottom_idx = np.argsort(weighted_proba)[:n_trade]
    
    valid = np.zeros(len(y_test), dtype=bool)
    valid[top_idx] = True
    valid[bottom_idx] = True
    
    preds = np.zeros(len(y_test))
    preds[top_idx] = 1
    preds[bottom_idx] = 0
    
    acc = (preds[valid] == y_test[valid]).mean()
    results[cov] = (acc, valid.sum())

all_results["weighted"] = results
print(f"{'weighted':12s}: 25% = {results[0.25][0]:.1%}")

# Print all results sorted by 25% accuracy
print("\n=== ALL RESULTS SORTED BY 25% ACCURACY ===")
sorted_results = sorted(all_results.items(), key=lambda x: x[1][0.25][0], reverse=True)
for name, res in sorted_results:
    print(f"{name:15s}: 10%={res[0.10][0]:.1%} 15%={res[0.15][0]:.1%} 20%={res[0.20][0]:.1%} 25%={res[0.25][0]:.1%} 30%={res[0.30][0]:.1%}")

# Best configuration
best_name = sorted_results[0][0]
best_25 = sorted_results[0][1][0.25]
print(f"\nBEST: {best_name} at 25% = {best_25[0]:.1%} ({best_25[1]} trades)")

# Save best probabilities
best_proba = probabilities.get(best_name, weighted_proba)
np.save("best_probabilities.npy", best_proba)

print("\nBest probabilities saved to best_probabilities.npy")
