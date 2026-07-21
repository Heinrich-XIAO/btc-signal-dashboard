"""Quick test of ML feasibility at 25% coverage."""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("btc_5m_features.csv", index_col="timestamp", parse_dates=True)
df_clean = df.dropna()

exclude = ["open", "high", "low", "close", "volume", "trades", 
           "taker_buy_volume", "quote_volume", "returns", "log_returns",
           "target", "hour", "day_of_week"]
feature_cols = [c for c in df_clean.columns if c not in exclude]

X = df_clean[feature_cols].values
y = df_clean["target"].values

# Split: train on first 70%, test on last 30%
split_idx = int(len(df_clean) * 0.7)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
print(f"Baseline accuracy: {y_test.mean():.3f}")
print()

# Test Logistic Regression
lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
lr.fit(X_train_s, y_train)
proba_lr = lr.predict_proba(X_test_s)[:, 1]

# Find accuracy at different coverage levels
for coverage in [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
    n_trade = int(len(X_test) * coverage)
    # Take top N confident up predictions and top N confident down predictions
    sorted_up = np.sort(proba_lr)[::-1]
    sorted_down = np.sort(proba_lr)
    
    # Method: threshold based
    # Find threshold such that we trade on ~coverage% of samples
    # Trade when proba > threshold_up (predict up) or proba < threshold_down (predict down)
    
    threshold_up = np.percentile(proba_lr, 100 - coverage * 50)
    threshold_down = np.percentile(proba_lr, coverage * 50)
    
    mask_up = proba_lr > threshold_up
    mask_down = proba_lr < threshold_down
    mask = mask_up | mask_down
    
    preds = np.where(mask_up, 1, np.where(mask_down, 0, -1))
    valid = preds != -1
    
    if valid.sum() > 0:
        acc = (preds[valid] == y_test[valid]).mean()
        print(f"LR  Coverage={coverage:.0%}: threshold_up={threshold_up:.3f}, down={threshold_down:.3f}, "
              f"trades={valid.sum()}, accuracy={acc:.3f}")

print()

# Test Random Forest (faster with fewer trees)
rf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
proba_rf = rf.predict_proba(X_test)[:, 1]

for coverage in [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
    threshold_up = np.percentile(proba_rf, 100 - coverage * 50)
    threshold_down = np.percentile(proba_rf, coverage * 50)
    
    mask_up = proba_rf > threshold_up
    mask_down = proba_rf < threshold_down
    mask = mask_up | mask_down
    
    preds = np.where(mask_up, 1, np.where(mask_down, 0, -1))
    valid = preds != -1
    
    if valid.sum() > 0:
        acc = (preds[valid] == y_test[valid]).mean()
        print(f"RF  Coverage={coverage:.0%}: threshold_up={threshold_up:.3f}, down={threshold_down:.3f}, "
              f"trades={valid.sum()}, accuracy={acc:.3f}")

print()

# Also test: what if we only trade on one side (only predict up when very confident)?
print("--- One-sided (Up only) ---")
for coverage in [0.10, 0.20, 0.25]:
    threshold = np.percentile(proba_lr, 100 - coverage)
    mask = proba_lr > threshold
    if mask.sum() > 0:
        acc = y_test[mask].mean()
        print(f"LR  Up-only coverage={coverage:.0%}: trades={mask.sum()}, accuracy={acc:.3f}")

print()
print("--- One-sided (Down only) ---")
for coverage in [0.10, 0.20, 0.25]:
    threshold = np.percentile(proba_lr, coverage)
    mask = proba_lr < threshold
    if mask.sum() > 0:
        acc = 1 - y_test[mask].mean()
        print(f"LR  Down-only coverage={coverage:.0%}: trades={mask.sum()}, accuracy={acc:.3f}")
