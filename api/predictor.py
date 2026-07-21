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
        self.combined_features = self.meta["combined_features"]
        self.original_features = self.meta["original_features"]
        self.ensemble_config = self.meta["ensemble_config"]

        # Load models
        self.lr_all = joblib.load(MODELS_DIR / "lr_all.joblib")
        self.lr_comb = joblib.load(MODELS_DIR / "lr_comb.joblib")
        self.scaler_all = joblib.load(MODELS_DIR / "scaler_all.joblib")
        self.scaler_comb = joblib.load(MODELS_DIR / "scaler_comb.joblib")

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

        # Thresholds from training
        self.up_thresh = self.ensemble_config["up_threshold"]
        self.down_thresh = self.ensemble_config["down_threshold"]

    def _get_features(self, df: pd.DataFrame, feature_list: list) -> np.ndarray:
        """Extract feature array, handling missing columns and NaNs gracefully."""
        available = [c for c in feature_list if c in df.columns]
        missing = [c for c in feature_list if c not in df.columns]
        if missing:
            print(f"Warning: missing features {missing}")
        X = df[available].fillna(0).values
        # If some features are missing, fill with zeros (not ideal but safe for live)
        if len(available) < len(feature_list):
            X_full = np.zeros((len(df), len(feature_list)))
            for i, col in enumerate(feature_list):
                if col in df.columns:
                    X_full[:, i] = df[col].fillna(0).values
            X = X_full
        return X

    def predict(self, df: pd.DataFrame) -> dict:
        """Run all models on the latest row of the dataframe."""
        if df is None or df.empty:
            return {"error": "No data"}

        latest = df.iloc[-1:].copy()

        # Get features
        X_all = self._get_features(latest, self.all_features)
        X_comb = self._get_features(latest, self.combined_features)

        # Scale
        X_all_s = self.scaler_all.transform(X_all)
        X_comb_s = self.scaler_comb.transform(X_comb)

        # Predictions (probability of UP)
        proba_lr_all = float(self.lr_all.predict_proba(X_all_s)[0, 1])
        proba_lr_comb = float(self.lr_comb.predict_proba(X_comb_s)[0, 1])
        proba_xgb = float(self.xgb.predict_proba(X_all)[0, 1])
        lgb_pred = self.lgb.predict(X_all)
        proba_lgb = float(lgb_pred[0] if hasattr(lgb_pred, '__len__') else lgb_pred)

        # Ensemble (equal weights)
        ensemble_proba = np.mean([proba_lr_comb, proba_xgb, proba_lgb])

        # Determine signal
        if ensemble_proba > self.up_thresh:
            signal = "UP"
            confidence = float((ensemble_proba - self.up_thresh) / (1 - self.up_thresh))
        elif ensemble_proba < self.down_thresh:
            signal = "DOWN"
            confidence = float((self.down_thresh - ensemble_proba) / self.down_thresh)
        else:
            signal = "HOLD"
            confidence = 0.0

        # Individual model votes
        models = {
            "lr_comb": {"proba": proba_lr_comb, "signal": self._signal(proba_lr_comb)},
            "xgb": {"proba": proba_xgb, "signal": self._signal(proba_xgb)},
            "lgb": {"proba": proba_lgb, "signal": self._signal(proba_lgb)},
            "lr_all": {"proba": proba_lr_all, "signal": self._signal(proba_lr_all)},
        }

        if self.cat is not None:
            proba_cat = float(self.cat.predict_proba(X_all)[0, 1])
            models["cat"] = {"proba": proba_cat, "signal": self._signal(proba_cat)}

        if self.rf is not None:
            proba_rf = float(self.rf.predict_proba(X_all)[0, 1])
            models["rf"] = {"proba": proba_rf, "signal": self._signal(proba_rf)}

        if self.hgb is not None:
            proba_hgb = float(self.hgb.predict_proba(X_all)[0, 1])
            models["hgb"] = {"proba": proba_hgb, "signal": self._signal(proba_hgb)}

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
