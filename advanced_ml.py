"""Advanced feature engineering + powerful ML models."""
import pandas as pd
import numpy as np
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("btc_5m_features.csv", index_col="timestamp", parse_dates=True)

print("Engineering advanced features...")

# Original features
exclude = ["open", "high", "low", "close", "volume", "trades", 
           "taker_buy_volume", "quote_volume", "returns", "log_returns",
           "target", "hour", "day_of_week"]
base_features = [c for c in df.columns if c not in exclude]

# Add interaction features
key_features = ["rsi_14", "stoch_k_14", "bb_20_position", "macd_hist",
                "close_ma20_dist", "volume_ratio", "taker_buy_ratio",
                "returns_lag1", "cumret_5", "williams_r_14", "cci_14"]

for i, f1 in enumerate(key_features):
    for f2 in key_features[i+1:]:
        df[f"{f1}_x_{f2}"] = df[f1] * df[f2]
        df[f"{f1}_div_{f2}"] = df[f1] / (df[f2] + 1e-8)

# Add rate-of-change features
for f in key_features:
    df[f"{f}_roc1"] = df[f].diff(1)
    df[f"{f}_roc3"] = df[f].diff(3)
    df[f"{f}_roc5"] = df[f].diff(5)

# Add normalized/rank features
for f in key_features:
    df[f"{f}_rank10"] = df[f].rolling(10).apply(lambda x: pd.Series(x).rank().iloc[-1] / len(x))
    df[f"{f}_zscore20"] = (df[f] - df[f].rolling(20).mean()) / df[f].rolling(20).std()

# Add categorical bins
for f in key_features:
    try:
        df[f"{f}_q10"] = pd.qcut(df[f], q=10, labels=False, duplicates="drop")
    except:
        pass

# Add time interactions
df["hour_x_rsi"] = df["hour"] * df["rsi_14"]
df["hour_x_volume"] = df["hour"] * df["volume_ratio"]

print(f"Total features: {len(df.columns)}")

# Drop rows with NaNs
df_clean = df.dropna()
print(f"Clean rows: {len(df_clean)}")

# Get all feature columns
all_feature_cols = [c for c in df_clean.columns if c not in exclude + ["target"]]
print(f"Feature columns: {len(all_feature_cols)}")

X = df_clean[all_feature_cols].values
y = df_clean["target"].values

# Walk-forward split
split_idx = int(len(df_clean) * 0.7)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"Train: {len(X_train)}, Test: {len(X_test)}")

# Scale for some models
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# Test HistGradientBoosting with different learning rates and max_iter
for lr in [0.01, 0.05, 0.1]:
    for max_iter in [100, 200, 500]:
        model = HistGradientBoostingClassifier(learning_rate=lr, max_iter=max_iter, 
                                                max_depth=5, random_state=42)
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        
        for coverage in [0.15, 0.20, 0.25, 0.30]:
            # Two-sided threshold
            threshold_up = np.percentile(proba, 100 - coverage * 50)
            threshold_down = np.percentile(proba, coverage * 50)
            
            mask_up = proba > threshold_up
            mask_down = proba < threshold_down
            valid = mask_up | mask_down
            
            if valid.sum() > 0:
                preds = np.where(mask_up, 1, 0)
                acc = (preds[valid] == y_test[valid]).mean()
                if acc > 0.60:
                    print(f"HGB lr={lr} iter={max_iter} coverage={coverage:.0%}: "
                          f"acc={acc:.3f}, trades={valid.sum()}")

# Also try: only take the most confident predictions (both sides combined)
print("\n=== Top-N selection approach ===")
for coverage in [0.10, 0.15, 0.20, 0.25]:
    n_select = int(len(X_test) * coverage / 2)
    
    # Top N confident up
    top_up_idx = np.argsort(proba)[-n_select:]
    # Top N confident down  
    top_down_idx = np.argsort(proba)[:n_select]
    
    preds = np.zeros(len(y_test))
    preds[top_up_idx] = 1
    preds[top_down_idx] = 0
    
    valid = np.zeros(len(y_test), dtype=bool)
    valid[top_up_idx] = True
    valid[top_down_idx] = True
    
    acc = (preds[valid] == y_test[valid]).mean()
    print(f"Top-N coverage={coverage:.0%}: trades={valid.sum()}, accuracy={acc:.3f}")

print("\nDone.")
