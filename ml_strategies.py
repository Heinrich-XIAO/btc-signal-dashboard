"""ML-based strategies to achieve 90% accuracy with 25% coverage."""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

def prepare_data(df, feature_cols=None):
    """Prepare X, y from dataframe."""
    df_clean = df.dropna()
    
    if feature_cols is None:
        # Exclude non-feature columns
        exclude = ["open", "high", "low", "close", "volume", "trades", 
                   "taker_buy_volume", "quote_volume", "returns", "log_returns",
                   "target", "hour", "day_of_week"]
        feature_cols = [c for c in df_clean.columns if c not in exclude]
    
    X = df_clean[feature_cols].values
    y = df_clean["target"].values
    
    return X, y, feature_cols, df_clean

def strategy_ml_threshold(df, model_class, model_params, feature_cols=None,
                          confidence_threshold=0.85, scaler=True, random_state=42):
    """
    ML strategy that trains a model and only trades when confidence > threshold.
    Uses walk-forward approach: train on first 70%, test on last 30%.
    """
    df_clean = df.dropna().copy()
    
    if feature_cols is None:
        exclude = ["open", "high", "low", "close", "volume", "trades", 
                   "taker_buy_volume", "quote_volume", "returns", "log_returns",
                   "target", "hour", "day_of_week"]
        feature_cols = [c for c in df_clean.columns if c not in exclude]
    
    X = df_clean[feature_cols].values
    y = df_clean["target"].values
    
    # Split: train on first 70%, test on last 30% (walk-forward)
    split_idx = int(len(df_clean) * 0.7)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    if scaler:
        s = StandardScaler()
        X_train = s.fit_transform(X_train)
        X_test = s.transform(X_test)
    
    model = model_class(random_state=random_state, **model_params)
    model.fit(X_train, y_train)
    
    # Get probabilities
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)[:, 1]
    else:
        # SVM decision function -> sigmoid
        decision = model.decision_function(X_test)
        proba = 1 / (1 + np.exp(-decision))
    
    # Create signal: only trade when very confident
    signal = np.zeros(len(X_test))
    signal[proba > confidence_threshold] = 1  # Very confident up
    signal[proba < (1 - confidence_threshold)] = -1  # Very confident down
    
    # Full signal array aligned with df_clean
    full_signal = np.zeros(len(df_clean))
    full_signal[:split_idx] = 0  # No trades in training period for evaluation
    full_signal[split_idx:] = signal
    
    df_clean["signal"] = full_signal
    return df_clean["signal"]

# ===== Strategy wrappers for the 10 ML models =====

def strategy_lr(df, confidence_threshold=0.85, C=1.0):
    return strategy_ml_threshold(df, LogisticRegression, {"C": C, "max_iter": 1000},
                                  confidence_threshold=confidence_threshold)

def strategy_rf(df, confidence_threshold=0.85, n_estimators=200, max_depth=10):
    return strategy_ml_threshold(df, RandomForestClassifier,
                                  {"n_estimators": n_estimators, "max_depth": max_depth},
                                  confidence_threshold=confidence_threshold)

def strategy_gb(df, confidence_threshold=0.85, n_estimators=200, max_depth=5):
    return strategy_ml_threshold(df, GradientBoostingClassifier,
                                  {"n_estimators": n_estimators, "max_depth": max_depth},
                                  confidence_threshold=confidence_threshold)

def strategy_svc(df, confidence_threshold=0.85, C=1.0):
    return strategy_ml_threshold(df, SVC, {"C": C, "probability": True},
                                  confidence_threshold=confidence_threshold, scaler=True)

def strategy_knn(df, confidence_threshold=0.85, n_neighbors=50):
    return strategy_ml_threshold(df, KNeighborsClassifier,
                                  {"n_neighbors": n_neighbors},
                                  confidence_threshold=confidence_threshold, scaler=True)

def strategy_adaboost(df, confidence_threshold=0.85, n_estimators=200):
    return strategy_ml_threshold(df, AdaBoostClassifier,
                                  {"n_estimators": n_estimators},
                                  confidence_threshold=confidence_threshold)

def strategy_extratrees(df, confidence_threshold=0.85, n_estimators=200, max_depth=10):
    return strategy_ml_threshold(df, ExtraTreesClassifier,
                                  {"n_estimators": n_estimators, "max_depth": max_depth},
                                  confidence_threshold=confidence_threshold)

def strategy_mlp(df, confidence_threshold=0.85, hidden_layer_sizes=(100, 50)):
    return strategy_ml_threshold(df, MLPClassifier,
                                  {"hidden_layer_sizes": hidden_layer_sizes, "max_iter": 500},
                                  confidence_threshold=confidence_threshold, scaler=True)

def strategy_nb(df, confidence_threshold=0.85):
    return strategy_ml_threshold(df, GaussianNB, {},
                                  confidence_threshold=confidence_threshold, scaler=True)

def strategy_gb_deep(df, confidence_threshold=0.85, n_estimators=300, max_depth=8):
    return strategy_ml_threshold(df, GradientBoostingClassifier,
                                  {"n_estimators": n_estimators, "max_depth": max_depth,
                                   "learning_rate": 0.05},
                                  confidence_threshold=confidence_threshold)

ML_STRATEGIES = {
    "logistic_regression": strategy_lr,
    "random_forest": strategy_rf,
    "gradient_boosting": strategy_gb,
    "svc": strategy_svc,
    "knn": strategy_knn,
    "adaboost": strategy_adaboost,
    "extra_trees": strategy_extratrees,
    "mlp": strategy_mlp,
    "naive_bayes": strategy_nb,
    "gradient_boosting_deep": strategy_gb_deep,
}

if __name__ == "__main__":
    from strategies import evaluate_signal
    df = pd.read_csv("btc_5m_features.csv", index_col="timestamp", parse_dates=True)
    
    print("Testing ML strategies with default parameters:\n")
    for name, func in ML_STRATEGIES.items():
        signal = func(df)
        df_test = df.dropna().copy()
        df_test["signal"] = signal
        # Only evaluate on test portion (last 30%)
        split_idx = int(len(df_test) * 0.7)
        df_test = df_test.iloc[split_idx:]
        result = evaluate_signal(df_test, "signal", min_trades=100)
        print(f"{name:25s}: accuracy={result['accuracy']:.3f}, trades={result['n_trades']}, "
              f"up={result['n_up']}, down={result['n_down']}")
