"""Train on March–June 1, test on June 1–July 1. Proper out-of-sample evaluation."""
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

DATA_PATH = Path("btc_5m_enhanced.csv")
MODELS_DIR = Path("models")

df = pd.read_csv(DATA_PATH, index_col="timestamp", parse_dates=True, date_format="ISO8601")
print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
print(f"Date range: {df.index[0]} to {df.index[-1]}")

# Drop rows with NaN target
df = df.dropna(subset=["target"])

# Shift all features by 1 (no lookahead)
feature_cols = [c for c in df.columns if c != "target"]
for col in feature_cols:
    df[col] = df[col].shift(1)
df = df.dropna()
print(f"After shift+dropna: {len(df)} rows")

# Date-based split
train = df[(df.index >= "2026-03-01") & (df.index < "2026-06-01")]
test = df[(df.index >= "2026-06-01") & (df.index < "2026-07-01")]
print(f"Train: {len(train)} rows ({train.index[0].date()} to {train.index[-1].date()})")
print(f"Test:  {len(test)} rows ({test.index[0].date()} to {test.index[-1].date()})")

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

if "funding_rate" in train.columns:
    all_features.extend(["funding_rate", "funding_rate_abs", "funding_rate_trend"])

all_features = [c for c in all_features if c in train.columns]
original_features = [c for c in original_features if c in train.columns]
combined_features = sorted(set(all_features + original_features))

print(f"all_features: {len(all_features)}, combined: {len(combined_features)}")

# Prepare arrays
X_all_train = train[all_features].values
X_comb_train = train[combined_features].values
y_train = train["target"].values

X_all_test = test[all_features].values
X_comb_test = test[combined_features].values
y_test = test["target"].values

# Scalers (fit on train only)
scaler_all = StandardScaler().fit(X_all_train)
scaler_comb = StandardScaler().fit(X_comb_train)

X_all_train_s = scaler_all.transform(X_all_train)
X_comb_train_s = scaler_comb.transform(X_comb_train)
X_all_test_s = scaler_all.transform(X_all_test)
X_comb_test_s = scaler_comb.transform(X_comb_test)

# === Train all models ===
print("\nTraining models...")

lr_all = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
lr_all.fit(X_all_train_s, y_train)
proba_lr_all_test = lr_all.predict_proba(X_all_test_s)[:, 1]

lr_comb = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
lr_comb.fit(X_comb_train_s, y_train)
proba_lr_comb_test = lr_comb.predict_proba(X_comb_test_s)[:, 1]

xgb_model = xgb.XGBClassifier(
    learning_rate=0.1, max_depth=3, n_estimators=500,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, n_jobs=-1, verbosity=0
)
xgb_model.fit(X_all_train, y_train)
proba_xgb_test = xgb_model.predict_proba(X_all_test)[:, 1]

lgb_model = lgb.LGBMClassifier(
    learning_rate=0.1, max_depth=5, n_estimators=200,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, n_jobs=-1, verbosity=-1
)
lgb_model.fit(X_all_train, y_train)
proba_lgb_test = lgb_model.predict_proba(X_all_test)[:, 1]

cat_model = CatBoostClassifier(
    iterations=500, depth=5, learning_rate=0.1,
    loss_function="Logloss", verbose=False, random_state=42
)
cat_model.fit(X_all_train, y_train)
proba_cat_test = cat_model.predict_proba(X_all_test)[:, 1]

rf_model = RandomForestClassifier(
    n_estimators=500, max_depth=10, min_samples_split=5,
    random_state=42, n_jobs=-1
)
rf_model.fit(X_all_train, y_train)
proba_rf_test = rf_model.predict_proba(X_all_test)[:, 1]

hgb_model = HistGradientBoostingClassifier(
    learning_rate=0.05, max_iter=200, max_depth=5, random_state=42
)
hgb_model.fit(X_all_train, y_train)
proba_hgb_test = hgb_model.predict_proba(X_all_test)[:, 1]

# === Ensemble probabilities on TEST set ===
ensemble_test = np.mean([proba_lr_comb_test, proba_xgb_test, proba_lgb_test], axis=0)

# === Thresholds from TRAINING set percentiles ===
# Also compute training ensemble for threshold setting
proba_lr_comb_train = lr_comb.predict_proba(X_comb_train_s)[:, 1]
proba_xgb_train = xgb_model.predict_proba(X_all_train)[:, 1]
proba_lgb_train = lgb_model.predict_proba(X_all_train)[:, 1]
ensemble_train = np.mean([proba_lr_comb_train, proba_xgb_train, proba_lgb_train], axis=0)

up_thresh = float(np.percentile(ensemble_train, 88))
down_thresh = float(np.percentile(ensemble_train, 12))

print(f"\nThresholds (from TRAINING set): UP>{up_thresh:.4f}, DOWN<{down_thresh:.4f}")
print(f"  (88th percentile = {up_thresh:.4f}, 12th percentile = {down_thresh:.4f})")

# === Evaluate on TEST set ===
def evaluate(name, proba, y_true, up_t, down_t):
    mask_up = proba > up_t
    mask_down = proba < down_t
    valid = mask_up | mask_down
    if valid.sum() == 0:
        return None
    preds = np.where(mask_up, 1, 0)
    acc = (preds[valid] == y_true[valid]).mean()
    tp = ((preds == 1) & (y_true == 1) & valid).sum()
    fp = ((preds == 1) & (y_true == 0) & valid).sum()
    tn = ((preds == 0) & (y_true == 0) & valid).sum()
    fn = ((preds == 0) & (y_true == 1) & valid).sum()
    return {
        "accuracy": float(acc),
        "coverage": float(valid.mean()),
        "n_trades": int(valid.sum()),
        "up_threshold": float(up_t),
        "down_threshold": float(down_t),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }

print("\n" + "=" * 80)
print("INDIVIDUAL MODEL RESULTS (test set, thresholds from TRAINING set)")
print("=" * 80)

models_data = {
    "lr_all": proba_lr_all_test,
    "lr_comb": proba_lr_comb_test,
    "xgb": proba_xgb_test,
    "lgb": proba_lgb_test,
    "cat": proba_cat_test,
    "rf": proba_rf_test,
    "hgb": proba_hgb_test,
}

results = {}
for name, proba in models_data.items():
    r = evaluate(name, proba, y_test, up_thresh, down_thresh)
    if r:
        results[name] = r
        print(f"  {name:10s}: {r['accuracy']:.1%} accuracy, {r['coverage']:.1%} coverage "
              f"({r['n_trades']} trades) TP={r['tp']} FP={r['fp']} TN={r['tn']} FN={r['fn']}")

# Ensemble
r_ens = evaluate("ensemble", ensemble_test, y_test, up_thresh, down_thresh)
results["ensemble"] = r_ens
print(f"\n  {'ENSEMBLE':10s}: {r_ens['accuracy']:.1%} accuracy, {r_ens['coverage']:.1%} coverage "
      f"({r_ens['n_trades']} trades) TP={r_ens['tp']} FP={r_ens['fp']} TN={r_ens['tn']} FN={r_ens['fn']}")

# === Also show what the OLD thresholds (0.58/0.47) would give ===
print("\n" + "=" * 80)
print("COMPARISON: Old thresholds (0.58/0.47) vs new (training-set calibrated)")
print("=" * 80)
r_old = evaluate("ensemble", ensemble_test, y_test, 0.58, 0.47)
print(f"  Old (0.58/0.47): {r_old['accuracy']:.1%} accuracy, {r_old['coverage']:.1%} coverage ({r_old['n_trades']} trades)")
print(f"  New ({up_thresh:.4f}/{down_thresh:.4f}): {r_ens['accuracy']:.1%} accuracy, {r_ens['coverage']:.1%} coverage ({r_ens['n_trades']} trades)")

# === Coverage breakdown ===
print("\n" + "=" * 80)
print("ACCURACY AT VARIOUS COVERAGE LEVELS (test set, thresholds from TRAINING set)")
print("=" * 80)
print(f"{'Pctile':>7} {'UP thr':>8} {'DN thr':>8} {'Trades':>7} {'Cov%':>6} {'Acc%':>6}")
for pct in [95, 92, 90, 88, 85, 80, 75]:
    lo = 100 - pct
    up_p = float(np.percentile(ensemble_train, pct))
    dn_p = float(np.percentile(ensemble_train, lo))
    r = evaluate("e", ensemble_test, y_test, up_p, dn_p)
    if r and r["n_trades"] >= 5:
        print(f"  {pct:>5}% {up_p:>8.4f} {dn_p:>8.4f} {r['n_trades']:>7} {r['coverage']*100:>5.1f}% {r['accuracy']*100:>5.1f}%")

# === Save models ===
print("\n" + "=" * 80)
print("SAVING MODELS")
print("=" * 80)

joblib.dump(lr_all, MODELS_DIR / "lr_all.joblib")
joblib.dump(lr_comb, MODELS_DIR / "lr_comb.joblib")
joblib.dump(scaler_all, MODELS_DIR / "scaler_all.joblib")
joblib.dump(scaler_comb, MODELS_DIR / "scaler_comb.joblib")
xgb_model.save_model(str(MODELS_DIR / "xgb.json"))
lgb_model.booster_.save_model(str(MODELS_DIR / "lgb.txt"))
cat_model.save_model(str(MODELS_DIR / "cat.cbm"))
joblib.dump(rf_model, MODELS_DIR / "rf.joblib")
joblib.dump(hgb_model, MODELS_DIR / "hgb.joblib")

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
    "training_rows": len(train),
    "test_rows": len(test),
    "train_date_range": f"{train.index[0].date()} to {train.index[-1].date()}",
    "test_date_range": f"{test.index[0].date()} to {test.index[-1].date()}",
    "note": "Date-based split. Thresholds set on TRAINING set percentiles. No data leakage.",
}

with open(MODELS_DIR / "metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print(f"Saved to {MODELS_DIR}/")
print(f"Thresholds: UP>{up_thresh:.4f}, DOWN<{down_thresh:.4f}")
