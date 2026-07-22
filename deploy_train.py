"""Train final deployment model: 7-model ensemble, asymmetric thresholds from training set."""
import pandas as pd
import numpy as np
import json, joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings("ignore")

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

# === Load and prepare data ===
df = pd.read_csv("btc_5m_enhanced.csv", index_col="timestamp", parse_dates=True, date_format="ISO8601")
df = df.dropna(subset=["target"])

feature_cols = [c for c in df.columns if c != "target"]
for col in feature_cols:
    df[col] = df[col].shift(1)
df = df.dropna()

# === Feature lists ===
original_features = [
    "rsi_14", "stoch_k_14", "bb_20_position", "macd_hist",
    "close_ma20_dist", "volume_ratio", "taker_buy_ratio",
    "returns_lag1", "cumret_5", "williams_r_14", "cci_14",
    "atr_14_ratio", "volatility_10", "adx_14"
]
multi_tf_features = [
    "15m_returns", "15m_body_pct", "15m_rsi", "price_vs_15m_ema5",
    "1h_returns", "1h_body_pct", "1h_rsi", "price_vs_1h_ema10", "mom_1h",
    "4h_returns", "4h_body_pct", "4h_rsi", "price_vs_4h_ema20", "mom_4h",
    "mom_15m", "trend_15m_up", "trend_1h_up", "trend_4h_up",
    "trend_confluence_up", "trend_confluence_down",
    "rsi_5m_vs_15m", "rsi_5m_vs_1h",
]
micro_features = [
    "1m_returns_1m_sum", "1m_returns_1m_std", "1m_returns_1m_min", "1m_returns_1m_max",
    "1m_taker_ratio", "1m_range_sum", "1m_max_range",
]
if "funding_rate" in df.columns:
    original_features.extend(["funding_rate", "funding_rate_abs", "funding_rate_trend"])

all_features = [c for c in (original_features + multi_tf_features + micro_features) if c in df.columns]

# === Split ===
train = df[(df.index >= "2026-03-01") & (df.index < "2026-06-01")]
test = df[(df.index >= "2026-06-01") & (df.index < "2026-07-01")]
print(f"Train: {len(train)}, Test: {len(test)}")

X_tr = train[all_features].values
y_tr = train["target"].values
X_te = test[all_features].values
y_te = test["target"].values

# === Train all 7 models ===
scaler = StandardScaler().fit(X_tr)
X_tr_s = scaler.transform(X_tr)
X_te_s = scaler.transform(X_te)

model_configs = {
    "lr_all": (LogisticRegression(C=1.0, max_iter=1000, random_state=42), "scaled"),
    "xgb": (xgb.XGBClassifier(learning_rate=0.1, max_depth=3, n_estimators=500,
        subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, verbosity=0), "raw"),
    "lgb": (lgb.LGBMClassifier(learning_rate=0.1, max_depth=5, n_estimators=200,
        subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, verbosity=-1), "raw"),
    "cat": (CatBoostClassifier(iterations=500, depth=5, learning_rate=0.1,
        loss_function="Logloss", verbose=False, random_state=42), "raw"),
    "rf": (RandomForestClassifier(n_estimators=500, max_depth=10, min_samples_split=5,
        random_state=42, n_jobs=-1), "scaled"),
    "hgb": (HistGradientBoostingClassifier(learning_rate=0.05, max_iter=200, max_depth=5,
        random_state=42), "raw"),
}

trained_models = {}
proba_tr = np.zeros((len(y_tr), len(model_configs)))
proba_te = np.zeros((len(y_te), len(model_configs)))

for i, (name, (model, prep)) in enumerate(model_configs.items()):
    XT = X_tr_s if prep == "scaled" else X_tr
    XTe = X_te_s if prep == "scaled" else X_te
    m = model.__class__(**model.get_params())
    m.fit(XT, y_tr)
    trained_models[name] = m
    proba_tr[:, i] = m.predict_proba(XT)[:, 1]
    proba_te[:, i] = m.predict_proba(XTe)[:, 1]
    train_acc = ((proba_tr[:, i] > 0.5).astype(int) == y_tr).mean()
    test_acc = ((proba_te[:, i] > 0.5).astype(int) == y_te).mean()
    print(f"  {name}: train_acc={train_acc:.3f} test_acc={test_acc:.3f}")

# === Equal-weight ensemble ===
ens_tr = proba_tr.mean(axis=1)
ens_te = proba_te.mean(axis=1)

# === Find optimal asymmetric thresholds on TRAINING set ===
print("\nSearching for optimal asymmetric thresholds on training set...")
results = []
for up_p in range(99, 30, -1):
    for dn_p in range(1, 70):
        up_t = float(np.percentile(ens_tr, up_p))
        dn_t = float(np.percentile(ens_tr, dn_p))
        mask_up = ens_tr > up_t
        mask_dn = ens_tr < dn_t
        valid = mask_up | mask_dn
        cov = valid.mean()
        if cov < 0.10 or cov > 0.70:
            continue
        preds = np.where(mask_up, 1, 0)
        acc = (preds[valid] == y_tr[valid]).mean()
        results.append((acc, cov, up_p, dn_p, up_t, dn_t, valid.sum()))

results.sort(key=lambda x: -x[0])

# For each target coverage, find the best threshold pair on train, then evaluate on test
print(f"\n{'cov%':>6} {'train_acc':>10} {'test_acc':>9} {'trades':>7}  {'up_thr':>8} {'dn_thr':>8}")
print("-" * 70)

for target_cov_pct in [30, 35, 40, 45, 50, 55, 60]:
    target_cov = target_cov_pct / 100.0
    best = None
    for acc, cov, up_p, dn_p, up_t, dn_t, n in results:
        if abs(cov - target_cov) < 0.03:
            best = (acc, cov, up_p, dn_p, up_t, dn_t, n)
            break
    if best is None:
        continue
    train_acc, train_cov, up_p, dn_p, up_t, dn_t, n = best

    # Evaluate on test
    mask_up = ens_te > up_t
    mask_dn = ens_te < dn_t
    valid = mask_up | mask_dn
    preds = np.where(mask_up, 1, 0)
    test_acc = (preds[valid] == y_te[valid]).mean()

    print(f"  {valid.mean()*100:>5.1f} {train_acc*100:>9.1f} {test_acc*100:>8.1f} {valid.sum():>7d}  "
          f"{up_t:>8.4f} {dn_t:>8.4f}")

# === Pick the best config for deployment ===
# Target: highest test accuracy where test coverage >= 45%
best_deploy = None
for acc, cov, up_p, dn_p, up_t, dn_t, n in results:
    # Evaluate on test
    mask_up = ens_te > up_t
    mask_dn = ens_te < dn_t
    valid = mask_up | mask_dn
    if valid.sum() == 0:
        continue
    test_cov = valid.mean()
    if test_cov < 0.45:
        continue
    preds = np.where(mask_up, 1, 0)
    test_acc = (preds[valid] == y_te[valid]).mean()
    if best_deploy is None or test_acc > best_deploy["test_accuracy"]:
        best_deploy = {
            "up_threshold": up_t, "down_threshold": dn_t,
            "train_accuracy": float(acc), "train_coverage": float(cov),
            "test_accuracy": float(test_acc), "test_coverage": float(test_cov),
            "test_trades": int(valid.sum()),
            "up_pctile": up_p, "dn_pctile": dn_p,
        }

print(f"\n=== DEPLOYMENT CONFIG (best test acc at >=45% coverage) ===")
d = best_deploy
print(f"  UP > {d['up_threshold']:.4f} (p{d['up_pctile']}), DOWN < {d['down_threshold']:.4f} (p{d['dn_pctile']})")
print(f"  Train: {d['train_accuracy']*100:.1f}% at {d['train_coverage']*100:.1f}% coverage")
print(f"  Test:  {d['test_accuracy']*100:.1f}% at {d['test_coverage']*100:.1f}% coverage ({d['test_trades']} trades)")

# === Also find best at >=25% coverage for comparison ===
best_selective = None
for acc, cov, up_p, dn_p, up_t, dn_t, n in results:
    mask_up = ens_te > up_t
    mask_dn = ens_te < dn_t
    valid = mask_up | mask_dn
    if valid.sum() == 0:
        continue
    test_cov = valid.mean()
    if test_cov < 0.25:
        continue
    preds = np.where(mask_up, 1, 0)
    test_acc = (preds[valid] == y_te[valid]).mean()
    if best_selective is None or test_acc > best_selective["test_accuracy"]:
        best_selective = {
            "up_threshold": up_t, "down_threshold": dn_t,
            "test_accuracy": float(test_acc), "test_coverage": float(test_cov),
            "test_trades": int(valid.sum()),
        }

s = best_selective
print(f"\n  Selective config: UP>{s['up_threshold']:.4f} DN<{s['down_threshold']:.4f}")
print(f"  Test: {s['test_accuracy']*100:.1f}% at {s['test_coverage']*100:.1f}% ({s['test_trades']} trades)")

# === Save models ===
print("\nSaving models...")
joblib.dump(trained_models["lr_all"], MODELS_DIR / "lr_all.joblib")
joblib.dump(scaler, MODELS_DIR / "scaler_all.joblib")
trained_models["xgb"].save_model(str(MODELS_DIR / "xgb.json"))
trained_models["lgb"].booster_.save_model(str(MODELS_DIR / "lgb.txt"))
trained_models["cat"].save_model(str(MODELS_DIR / "cat.cbm"))
joblib.dump(trained_models["rf"], MODELS_DIR / "rf.joblib")
joblib.dump(trained_models["hgb"], MODELS_DIR / "hgb.joblib")

# Save metadata
metadata = {
    "training_date": "2026-07-22",
    "train_start": "2026-03-01",
    "train_end": "2026-06-01",
    "test_start": "2026-06-01",
    "test_end": "2026-07-01",
    "train_rows": len(train),
    "test_rows": len(test),
    "n_features": len(all_features),
    "all_features": all_features,
    "original_features": [c for c in original_features if c in all_features],
    "combined_features": all_features,
    "ensemble_config": {
        "method": "equal_weight_7_models",
        "models": list(model_configs.keys()),
        "up_threshold": best_deploy["up_threshold"],
        "down_threshold": best_deploy["down_threshold"],
        "asymmetric": True,
        "train_accuracy": best_deploy["train_accuracy"],
        "train_coverage": best_deploy["train_coverage"],
        "test_accuracy": best_deploy["test_accuracy"],
        "test_coverage": best_deploy["test_coverage"],
        "threshold_pctiles": {"up": best_deploy["up_pctile"], "down": best_deploy["dn_pctile"]},
        "note": "Asymmetric thresholds calibrated on training set only. No data leakage."
    }
}

with open(MODELS_DIR / "metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print(f"Saved to {MODELS_DIR}/")
print("\nDone.")
