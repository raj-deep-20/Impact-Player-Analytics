"""
IPL Impact Player Analytics — Dual ML Models
1. Impact Player Substitution Classifier (GradientBoosting ≈ XGBoost)
2. Fantasy Points Regressor (GradientBoosting)
3. SHAP-style permutation explainability
"""

import numpy as np
import pandas as pd
import json
import pickle
from pathlib import Path

from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report,
    mean_absolute_error, r2_score
)
from sklearn.inspection import permutation_importance

from feature_engineering import FEATURE_COLS_CLASSIFIER, FEATURE_COLS_REGRESSOR


MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)


# ── 1. Impact Player Substitution Classifier ─────────────────────────────────

class ImpactPlayerClassifier:
    """
    Predicts whether a substitution at the current match state will lift
    win probability by >5%.  Binary classification.
    """
    
    def __init__(self):
        self.model = GradientBoostingClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.85,
            min_samples_leaf=15,
            random_state=42,
        )
        self.scaler       = StandardScaler()
        self.feature_cols = FEATURE_COLS_CLASSIFIER
        self.feature_importance_  = {}
        self.cv_scores_           = []
        self.trained_             = False
    
    def fit(self, df: pd.DataFrame):
        # Filter to Impact-rule matches only for training the sub-signal
        sub_df = df[df["has_impact_rule"] == 1].copy()
        
        # Upsample positive class (class imbalance)
        pos = sub_df[sub_df["optimal_sub"] == 1]
        neg = sub_df[sub_df["optimal_sub"] == 0].sample(
            n=min(len(neg_df := sub_df[sub_df["optimal_sub"]==0]), len(pos)*4),
            random_state=42
        )
        balanced = pd.concat([pos, neg]).sample(frac=1, random_state=42)
        
        X = balanced[self.feature_cols].fillna(0)
        y = balanced["optimal_sub"]
        
        X_scaled = self.scaler.fit_transform(X)
        
        # Cross-validation
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_aucs = cross_val_score(self.model, X_scaled, y, cv=skf, scoring="roc_auc")
        self.cv_scores_ = cv_aucs
        
        self.model.fit(X_scaled, y)
        self.trained_ = True
        
        # Feature importance from model
        importances = self.model.feature_importances_
        self.feature_importance_ = dict(sorted(
            zip(self.feature_cols, importances),
            key=lambda x: x[1], reverse=True
        ))
        
        return self
    
    def predict_proba(self, state_dict: dict) -> dict:
        """
        state_dict: match state features → returns substitution recommendation.
        """
        row = pd.DataFrame([state_dict])[self.feature_cols].fillna(0)
        X   = self.scaler.transform(row)
        proba = self.model.predict_proba(X)[0]
        
        return {
            "sub_probability":   float(proba[1]),
            "no_sub_probability":float(proba[0]),
            "recommend_sub":     bool(proba[1] > 0.52),
        }
    
    def explain(self, state_dict: dict, top_n: int = 6) -> list[dict]:
        """SHAP-style permutation explanation for a single prediction."""
        row      = pd.DataFrame([state_dict])[self.feature_cols].fillna(0)
        X_base   = self.scaler.transform(row)
        baseline = self.model.predict_proba(X_base)[0][1]
        
        contributions = []
        for i, feat in enumerate(self.feature_cols):
            X_perm    = X_base.copy()
            X_perm[0, i] = 0  # zero-out the feature
            perturbed = self.model.predict_proba(X_perm)[0][1]
            delta     = baseline - perturbed
            
            val = state_dict.get(feat, 0)
            contributions.append({
                "feature":     feat,
                "value":       round(float(val), 3),
                "shap_value":  round(float(delta), 4),
                "direction":   "↑" if delta > 0 else "↓",
                "importance":  self.feature_importance_.get(feat, 0),
            })
        
        contributions.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
        return contributions[:top_n]
    
    def save(self, path: str = "models/impact_classifier.pkl"):
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "scaler": self.scaler,
                         "feature_importance": self.feature_importance_,
                         "cv_scores": self.cv_scores_.tolist()}, f)
        print(f"✓ Classifier saved → {path}")
    
    def load(self, path: str = "models/impact_classifier.pkl"):
        with open(path, "rb") as f:
            obj = pickle.load(f)
        self.model               = obj["model"]
        self.scaler              = obj["scaler"]
        self.feature_importance_ = obj["feature_importance"]
        self.cv_scores_          = obj["cv_scores"]
        self.trained_            = True


# ── 2. Fantasy Points Regressor ───────────────────────────────────────────────

class FantasyPointsRegressor:
    """
    Predicts expected fantasy points for a player in the upcoming match.
    Used to rank top-5 picks with confidence scores.
    """
    
    def __init__(self):
        self.model = GradientBoostingRegressor(
            n_estimators=250,
            max_depth=4,
            learning_rate=0.10,
            subsample=0.80,
            min_samples_leaf=10,
            random_state=42,
        )
        self.scaler        = StandardScaler()
        self.feature_cols  = FEATURE_COLS_REGRESSOR
        self.feature_importance_ = {}
        self.cv_mae_       = []
        self.trained_      = False
    
    def fit(self, df: pd.DataFrame):
        df_clean = df.dropna(subset=self.feature_cols + ["fantasy_pts_avg"])
        X = df_clean[self.feature_cols].fillna(0)
        y = df_clean["fantasy_pts_avg"]
        
        X_scaled = self.scaler.fit_transform(X)
        
        kf   = KFold(n_splits=5, shuffle=True, random_state=42)
        maes = cross_val_score(self.model, X_scaled, y, cv=kf,
                               scoring="neg_mean_absolute_error")
        self.cv_mae_ = (-maes).tolist()
        
        self.model.fit(X_scaled, y)
        self.trained_ = True
        
        importances = self.model.feature_importances_
        self.feature_importance_ = dict(sorted(
            zip(self.feature_cols, importances),
            key=lambda x: x[1], reverse=True
        ))
        
        return self
    
    def predict_player(self, player_row: dict) -> dict:
        """Predict fantasy points for one player row."""
        row      = pd.DataFrame([player_row])[self.feature_cols].fillna(0)
        X_scaled = self.scaler.transform(row)
        pred_pts = float(self.model.predict(X_scaled)[0])
        
        # Confidence: inverse of CV MAE relative to prediction
        mean_mae    = np.mean(self.cv_mae_)
        confidence  = max(0, min(1, 1 - mean_mae / max(pred_pts, 1)))
        
        return {
            "predicted_pts": round(pred_pts, 1),
            "confidence":    round(confidence, 3),
        }
    
    def rank_players(self, player_features_df: pd.DataFrame,
                     top_n: int = 5) -> pd.DataFrame:
        """Return ranked top-N fantasy picks for a given match."""
        df_c = player_features_df.copy()
        
        predictions = []
        for _, row in df_c.iterrows():
            pred = self.predict_player(row.to_dict())
            predictions.append(pred)
        
        pred_df = pd.DataFrame(predictions)
        result  = pd.concat([df_c.reset_index(drop=True), pred_df], axis=1)
        result  = result.sort_values("predicted_pts", ascending=False)
        
        return result.head(top_n)
    
    def explain(self, player_row: dict, top_n: int = 5) -> list[dict]:
        """SHAP-style explanation for a player prediction."""
        row      = pd.DataFrame([player_row])[self.feature_cols].fillna(0)
        X_base   = self.scaler.transform(row)
        baseline = float(self.model.predict(X_base)[0])
        
        contributions = []
        for i, feat in enumerate(self.feature_cols):
            X_perm       = X_base.copy()
            X_perm[0, i] = 0
            perturbed    = float(self.model.predict(X_perm)[0])
            delta        = baseline - perturbed
            
            contributions.append({
                "feature":    feat,
                "value":      round(float(row.iloc[0, i]), 3),
                "shap_value": round(delta, 2),
                "direction":  "↑" if delta > 0 else "↓",
            })
        
        contributions.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
        return contributions[:top_n]
    
    def save(self, path: str = "models/fantasy_regressor.pkl"):
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "scaler": self.scaler,
                         "feature_importance": self.feature_importance_,
                         "cv_mae": self.cv_mae_}, f)
        print(f"✓ Regressor  saved → {path}")
    
    def load(self, path: str = "models/fantasy_regressor.pkl"):
        with open(path, "rb") as f:
            obj = pickle.load(f)
        self.model               = obj["model"]
        self.scaler              = obj["scaler"]
        self.feature_importance_ = obj["feature_importance"]
        self.cv_mae_             = obj["cv_mae"]
        self.trained_            = True


# ── 3. Training Pipeline ─────────────────────────────────────────────────────

def train_all(snapshot_df: pd.DataFrame, player_df: pd.DataFrame) -> dict:
    """
    Train both models and return evaluation metrics.
    """
    print("\n" + "="*55)
    print("  IPL IMPACT PLAYER — MODEL TRAINING")
    print("="*55)
    
    # ── Classifier ──────────────────────────────────────────
    print("\n[1/2] Training Impact Player Classifier...")
    clf = ImpactPlayerClassifier()
    clf.fit(snapshot_df)
    clf.save()
    
    acc_cv = clf.cv_scores_.mean()
    print(f"  CV ROC-AUC  : {acc_cv:.4f} ± {clf.cv_scores_.std():.4f}")
    print(f"  Top features: {list(clf.feature_importance_.keys())[:5]}")
    
    # ── Regressor ────────────────────────────────────────────
    print("\n[2/2] Training Fantasy Points Regressor...")
    reg = FantasyPointsRegressor()
    reg.fit(player_df)
    reg.save()
    
    mean_mae = np.mean(reg.cv_mae_)
    print(f"  CV MAE      : {mean_mae:.2f} pts")
    print(f"  Top features: {list(reg.feature_importance_.keys())[:5]}")
    
    metrics = {
        "classifier": {
            "cv_roc_auc_mean": round(float(clf.cv_scores_.mean()), 4),
            "cv_roc_auc_std":  round(float(clf.cv_scores_.std()),  4),
            "top_features":    list(clf.feature_importance_.keys())[:8],
            "feature_importance": {k: round(float(v), 5)
                                   for k, v in clf.feature_importance_.items()},
        },
        "regressor": {
            "cv_mae_mean":  round(float(np.mean(reg.cv_mae_)), 2),
            "cv_mae_std":   round(float(np.std(reg.cv_mae_)),  2),
            "top_features": list(reg.feature_importance_.keys())[:8],
            "feature_importance": {k: round(float(v), 5)
                                   for k, v in reg.feature_importance_.items()},
        },
    }
    
    with open("models/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    print("\n✓ Training complete. Metrics saved → models/metrics.json")
    print("="*55)
    
    return metrics, clf, reg
