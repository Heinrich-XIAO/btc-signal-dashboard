"""Test advanced gradient boosting models on multi-timeframe features."""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("btc_5m_enhanced.csv", index_col="timestamp", parse_dates=True, date_format='ISO8601')
df_clean = df.dropna()

# Select the best features
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

if "funding_rate" in df_clean.columns:
    feature_cols.extend(["funding_rate", "funding_rate_abs", "funding_rate_trend"])

feature_cols = [c for c in feature_cols if c in df_clean.columns]
print(f"Using {len(feature_cols)} features")

X = df_clean[feature_cols].values
y = df_clean["target"].values

split_idx = int(len(df_clean) * 0.7)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

def evaluate_coverage(proba, y_true, coverages=[0.10, 0.15, 0.20, 0.25, 0.30]):
    results = []
    for cov in coverages:
        n_trade = int(len(y_true) * cov)
        top_idx = np.argsort(proba)[-n_trade:]
        bottom_idx = np.argsort(proba)[:n_trade]
        
        valid = np.zeros(len(y_true), dtype=bool)
        valid[top_idx] = True
        valid[bottom_idx] = True
        
        preds = np.zeros(len(y_true))
        preds[top_idx] = 1
        preds[bottom_idx] = 0
        
        acc = (preds[valid] == y_true[valid]).mean()
        results.append((cov, valid.sum(), acc))
    return results

print("\n=== XGBoost ===")
for lr in [0.01, 0.05, 0.1]:
    for depth in [3, 5, 7]:
        for n_est in [100, 200, 500]:
            model = xgb.XGBClassifier(
                learning_rate=lr, max_depth=depth, n_estimators=n_est,
                subsample=0.8, colsample_bytree=0.8,
                objective='binary:logistic', eval_metric='logloss',
                random_state=42, n_jobs=-1, verbosity=0
            )
            model.fit(X_train, y_train)
            proba = model.predict_proba(X_test)[:, 1]
            results = evaluate_coverage(proba, y_test)
            cov25_acc = [r[2] for r in results if r[0] == 0.25][0]
            if cov25_acc > 0.82:
                print(f"  lr={lr} depth={depth} n={n_est}: 25% cov = {cov25_acc:.1%}")
                for cov, trades, acc in results:
                    print(f"    {cov:>5.0%}: {acc:.1%} ({trades} trades)")

print("\n=== LightGBM ===")
for lr in [0.01, 0.05, 0.1]:
    for depth in [3, 5, 7]:
        for n_est in [100, 200, 500]:
            model = lgb.LGBMClassifier(
                learning_rate=lr, max_depth=depth, n_estimators=n_est,
                subsample=0.8, colsample_bytree=0.8,
                random_state=42, n_jobs=-1, verbosity=-1
            )
            model.fit(X_train, y_train)
            proba = model.predict_proba(X_test)[:, 1]
            results = evaluate_coverage(proba, y_test)
            cov25_acc = [r[2] for r in results if r[0] == 0.25][0]
            if cov25_acc > 0.82:
                print(f"  lr={lr} depth={depth} n={n_est}: 25% cov = {cov25_acc:.1%}")
                for cov, trades, acc in results:
                    print(f"    {cov:>5.0%}: {acc:.1%} ({trades} trades)")

print("\n=== CatBoost ===")
for depth in [3, 5, 7]:
    for iters in [100, 200, 500]:
        model = CatBoostClassifier(
            iterations=iters, depth=depth, learning_rate=0.1,
            loss_function='Logloss',
            random_seed=42, verbose=False
        )
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        results = evaluate_coverage(proba, y_test)
        cov25_acc = [r[2] for r in results if r[0] == 0.25][0]
        if cov25_acc > 0.80:
            print(f"  depth={depth} iters={iters}: 25% cov = {cov25_acc:.1%}")
            for cov, trades, acc in results:
                print(f"    {cov:>5.0%}: {acc:.1%} ({trades} trades)")

print("\nDone.")
