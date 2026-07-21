"""Save trained ML models to disk for live prediction API."""
import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings("ignore")

# Paths
DATA_PATH = Path("btc_5m_enhanced.csv")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH, index_col="timestamp", parse_dates=True, date_format='ISO8601')
df_clean = df.dropna()
print(f"Loaded {len(df_clean)} rows, {len(df_clean.columns)} columns")

# Feature sets (must match final_ultimate_v2.py exactly)
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

print(f"all_features: {len(all_features)}")
print(f"original_features: {len(original_features)}")
print(f"combined_features: {len(combined_features)}")

X_all = df_clean[all_features].values
X_comb = df_clean[combined_features].values
y = df_clean["target"].values
split_idx = int(len(df_clean) * 0.7)

# Train on full data (we want the best models for live prediction)
# But compute thresholds on the last 30% to match the paper
X_all_train, X_all_test = X_all[:split_idx], X_all[split_idx:]
X_comb_train, X_comb_test = X_comb[:split_idx], X_comb[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Scalers
scaler_all = StandardScaler()
X_all_train_s = scaler_all.fit_transform(X_all_train)

scaler_comb = StandardScaler()
X_comb_train_s = scaler_comb.fit_transform(X_comb_train)

# === Model 1: LR All ===
print("\nTraining LR All...")
lr_all = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
lr_all.fit(X_all_train_s, y_train)
proba_lr_all = lr_all.predict_proba(scaler_all.transform(X_all_test))[:, 1]

# === Model 2: LR Combined ===
print("Training LR Combined...")
lr_comb = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
lr_comb.fit(X_comb_train_s, y_train)
proba_lr_comb = lr_comb.predict_proba(scaler_comb.transform(X_comb_test))[:, 1]

# === Model 3: XGBoost ===
print("Training XGBoost...")
xgb_model = xgb.XGBClassifier(
    learning_rate=0.1, max_depth=3, n_estimators=500,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, n_jobs=-1, verbosity=0
)
xgb_model.fit(X_all_train, y_train)
proba_xgb = xgb_model.predict_proba(X_all_test)[:, 1]

# === Model 4: LightGBM ===
print("Training LightGBM...")
lgb_model = lgb.LGBMClassifier(
    learning_rate=0.1, max_depth=5, n_estimators=200,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, n_jobs=-1, verbosity=-1
)
lgb_model.fit(X_all_train, y_train)
proba_lgb = lgb_model.predict_proba(X_all_test)[:, 1]

# === Model 5: CatBoost ===
print("Training CatBoost...")
cat_model = CatBoostClassifier(
    iterations=500, depth=5, learning_rate=0.1,
    loss_function='Logloss', verbose=False, random_state=42
)
cat_model.fit(X_all_train, y_train)
proba_cat = cat_model.predict_proba(X_all_test)[:, 1]

# === Model 6: Random Forest ===
print("Training Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=500, max_depth=10, min_samples_split=5,
    random_state=42, n_jobs=-1
)
rf_model.fit(X_all_train, y_train)
proba_rf = rf_model.predict_proba(X_all_test)[:, 1]

# === Model 7: HistGradientBoosting ===
print("Training HistGradientBoosting...")
hgb_model = HistGradientBoostingClassifier(
    learning_rate=0.05, max_iter=200, max_depth=5, random_state=42
)
hgb_model.fit(X_all_train, y_train)
proba_hgb = hgb_model.predict_proba(X_all_test)[:, 1]

# Save models
print("\nSaving models...")
joblib.dump(lr_all, MODELS_DIR / "lr_all.joblib")
joblib.dump(lr_comb, MODELS_DIR / "lr_comb.joblib")
joblib.dump(scaler_all, MODELS_DIR / "scaler_all.joblib")
joblib.dump(scaler_comb, MODELS_DIR / "scaler_comb.joblib")
xgb_model.save_model(str(MODELS_DIR / "xgb.json"))
lgb_model.booster_.save_model(str(MODELS_DIR / "lgb.txt"))
cat_model.save_model(str(MODELS_DIR / "cat.cbm"))
joblib.dump(rf_model, MODELS_DIR / "rf.joblib")
joblib.dump(hgb_model, MODELS_DIR / "hgb.joblib")

# Compute ensemble and thresholds
ensemble_proba = np.mean([proba_lr_comb, proba_xgb, proba_lgb], axis=0)
up_thresh = float(np.percentile(ensemble_proba, 88))
down_thresh = float(np.percentile(ensemble_proba, 12))

# Evaluate all models
models_data = {
    "lr_all": {"proba": proba_lr_all, "scaler": "all"},
    "lr_comb": {"proba": proba_lr_comb, "scaler": "comb"},
    "xgb": {"proba": proba_xgb, "scaler": None},
    "lgb": {"proba": proba_lgb, "scaler": None},
    "cat": {"proba": proba_cat, "scaler": None},
    "rf": {"proba": proba_rf, "scaler": None},
    "hgb": {"proba": proba_hgb, "scaler": None},
}

# Evaluate at 24% coverage (12% each side)
results = {}
for name, data in models_data.items():
    proba = data["proba"]
    up_t = np.percentile(proba, 88)
    down_t = np.percentile(proba, 12)
    mask_up = proba > up_t
    mask_down = proba < down_t
    valid = mask_up | mask_down
    if valid.sum() > 0:
        preds = np.where(mask_up, 1, 0)
        acc = (preds[valid] == y_test[valid]).mean()
        results[name] = {
            "accuracy": float(acc),
            "coverage": float(valid.mean()),
            "n_trades": int(valid.sum()),
            "up_threshold": float(up_t),
            "down_threshold": float(down_t),
        }
        print(f"{name}: {acc:.1%} accuracy, {valid.mean():.1%} coverage ({valid.sum()} trades)")

# Ensemble results
mask_up_e = ensemble_proba > up_thresh
mask_down_e = ensemble_proba < down_thresh
valid_e = mask_up_e | mask_down_e
preds_e = np.where(mask_up_e, 1, 0)
acc_e = (preds_e[valid_e] == y_test[valid_e]).mean()

results["ensemble"] = {
    "accuracy": float(acc_e),
    "coverage": float(valid_e.mean()),
    "n_trades": int(valid_e.sum()),
    "up_threshold": up_thresh,
    "down_threshold": down_thresh,
    "models": ["lr_comb", "xgb", "lgb"],
}
print(f"\nENSEMBLE: {acc_e:.1%} accuracy, {valid_e.mean():.1%} coverage ({valid_e.sum()} trades)")

# Save metadata
metadata = {
    "all_features": all_features,
    "combined_features": combined_features,
    "original_features": original_features,
    "feature_count": len(combined_features),
    "model_results": results,
    "ensemble_config": {
        "models": ["lr_comb", "xgb", "lgb"],
        "weights": [1/3, 1/3, 1/3],
        "up_threshold": up_thresh,
        "down_threshold": down_thresh,
    },
    "training_rows": len(df_clean),
    "train_split": split_idx,
    "test_split": len(df_clean) - split_idx,
}

with open(MODELS_DIR / "metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print(f"\nAll models saved to {MODELS_DIR}/")
print(f"Metadata saved to {MODELS_DIR}/metadata.json")
