"""Load saved models and run live predictions."""
import json
import joblib
import numpy as np
from pathlib import Path
import pandas as pd

MODELS_DIR = Path(__file__).parent.parent / "models"


class Predictor:
    def __init__(self):
        with open(MODELS_DIR / "metadata.json", "r") as f:
            self.meta = json.load(f)

        self.all_features = self.meta["all_features"]
        self.ensemble_config = self.meta["ensemble_config"]

        # Single feature set (all features)
        self.features = self.meta["all_features"]

        # Load scaler
        self.scaler = joblib.load(MODELS_DIR / "scaler_all.joblib")

        # Load all 7 models
        self.lr_all = joblib.load(MODELS_DIR / "lr_all.joblib")

        import xgboost as xgb
        self.xgb = xgb.XGBClassifier()
        self.xgb.load_model(str(MODELS_DIR / "xgb.json"))

        import lightgbm as lgb
        self.lgb = lgb.Booster(model_file=str(MODELS_DIR / "lgb.txt"))

        self.cat = None
        if (MODELS_DIR / "cat.cbm").exists():
            from catboost import CatBoostClassifier
            self.cat = CatBoostClassifier()
            self.cat.load_model(str(MODELS_DIR / "cat.cbm"))

        self.rf = None
        if (MODELS_DIR / "rf.joblib").exists():
            self.rf = joblib.load(MODELS_DIR / "rf.joblib")

        self.hgb = None
        if (MODELS_DIR / "hgb.joblib").exists():
            self.hgb = joblib.load(MODELS_DIR / "hgb.joblib")

        # Asymmetric thresholds (calibrated on training set)
        self.up_thresh = self.ensemble_config["up_threshold"]
        self.down_thresh = self.ensemble_config["down_threshold"]

        # Model names for iteration
        self.model_names = ["lr_all", "xgb", "lgb", "cat", "rf", "hgb"]

    def _get_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract feature array, handling missing columns and NaNs gracefully."""
        available = [c for c in self.features if c in df.columns]
        missing = [c for c in self.features if c not in df.columns]
        if missing:
            print(f"Warning: missing features {missing}")

        if len(available) < len(self.features):
            X_full = np.zeros((len(df), len(self.features)))
            for i, col in enumerate(self.features):
                if col in df.columns:
                    X_full[:, i] = df[col].fillna(0).values
            return X_full

        return df[available].fillna(0).values

    def predict(self, df: pd.DataFrame) -> dict:
        """Run all models on the latest row of the dataframe."""
        if df is None or df.empty:
            return {"error": "No data"}

        latest = df.iloc[-1:].copy()

        # Get features (single feature set for all models)
        X = self._get_features(latest)
        X_scaled = self.scaler.transform(X)

        # Get probabilities from all models
        models = {}

        proba_lr = float(self.lr_all.predict_proba(X_scaled)[0, 1])
        models["lr_all"] = {"proba": proba_lr, "signal": self._signal(proba_lr)}

        proba_xgb = float(self.xgb.predict_proba(X)[0, 1])
        models["xgb"] = {"proba": proba_xgb, "signal": self._signal(proba_xgb)}

        lgb_pred = self.lgb.predict(X)
        proba_lgb = float(lgb_pred[0] if hasattr(lgb_pred, '__len__') else lgb_pred)
        models["lgb"] = {"proba": proba_lgb, "signal": self._signal(proba_lgb)}

        if self.cat is not None:
            proba_cat = float(self.cat.predict_proba(X)[0, 1])
            models["cat"] = {"proba": proba_cat, "signal": self._signal(proba_cat)}

        if self.rf is not None:
            proba_rf = float(self.rf.predict_proba(X_scaled)[0, 1])
            models["rf"] = {"proba": proba_rf, "signal": self._signal(proba_rf)}

        if self.hgb is not None:
            proba_hgb = float(self.hgb.predict_proba(X)[0, 1])
            models["hgb"] = {"proba": proba_hgb, "signal": self._signal(proba_hgb)}

        # Equal-weight ensemble of all available models
        probas = [m["proba"] for m in models.values()]
        ensemble_proba = np.mean(probas)

        # Determine signal using asymmetric thresholds
        if ensemble_proba > self.up_thresh:
            signal = "UP"
            confidence = float((ensemble_proba - self.up_thresh) / (1 - self.up_thresh))
        elif ensemble_proba < self.down_thresh:
            signal = "DOWN"
            confidence = float((self.down_thresh - ensemble_proba) / self.down_thresh)
        else:
            signal = "HOLD"
            confidence = 0.0

        # Count votes
        up_votes = sum(1 for m in models.values() if m["signal"] == "UP")
        down_votes = sum(1 for m in models.values() if m["signal"] == "DOWN")
        total = len(models)

        return {
            "signal": signal,
            "confidence": round(confidence * 100, 1),
            "ensemble_proba": round(ensemble_proba * 100, 2),
            "up_threshold": round(self.up_thresh * 100, 2),
            "down_threshold": round(self.down_thresh * 100, 2),
            "up_votes": up_votes,
            "down_votes": down_votes,
            "total_models": total,
            "models": {k: {"proba": round(v["proba"] * 100, 2), "signal": v["signal"]} for k, v in models.items()},
        }

    def _signal(self, proba: float) -> str:
        if proba > self.up_thresh:
            return "UP"
        elif proba < self.down_thresh:
            return "DOWN"
        return "HOLD"
