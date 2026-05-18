"""
models.py
---------
Training, evaluation, and serialization of four cricket ML models:
  1. Match winner classifier (pre-match)
  2. First innings score regressor
  3. Live win probability (ball-by-ball LightGBM)
  4. Player of the Match classifier

All models use a consistent API:
    model, metrics = train_<model>(X, y)
    joblib.dump(model, "models/<name>.pkl")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
import lightgbm as lgb
import xgboost as xgb

logger = logging.getLogger(__name__)
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
MODELS_DIR.mkdir(exist_ok=True)

MetricsDict = Dict[str, Any]


# ---------------------------------------------------------------------------
# 1. Match Winner Classifier
# ---------------------------------------------------------------------------


def train_match_winner(
    feature_df: pd.DataFrame,
    target_col: str = "team1_won",
    save: bool = True,
) -> Tuple[Any, MetricsDict]:
    """
    XGBoost classifier predicting whether team1 wins.

    Parameters
    ----------
    feature_df : output of features.build_match_features()
    target_col : binary target column name
    save       : whether to persist the model to disk

    Returns
    -------
    (model, metrics_dict)
    """
    X = feature_df.drop(columns=[target_col])
    y = feature_df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    proba = model.predict_proba(X_test)[:, 1]
    preds = model.predict(X_test)

    cv_scores = cross_val_score(model, X, y, cv=StratifiedKFold(5), scoring="roc_auc")

    metrics = {
        "accuracy": round(accuracy_score(y_test, preds), 4),
        "roc_auc": round(roc_auc_score(y_test, proba), 4),
        "log_loss": round(log_loss(y_test, proba), 4),
        "cv_roc_auc_mean": round(cv_scores.mean(), 4),
        "cv_roc_auc_std": round(cv_scores.std(), 4),
        "feature_importances": dict(zip(X.columns, model.feature_importances_)),
    }

    logger.info(
        "Match winner model — AUC: %.4f | Acc: %.4f",
        metrics["roc_auc"],
        metrics["accuracy"],
    )

    if save:
        # Save using XGBoost's native binary JSON format (.ubj).
        # joblib pickle is XGBoost-version-sensitive and triggers a
        # UserWarning on every load when the XGBoost version changes.
        # .ubj is the stable, version-portable serialization format.
        model.save_model(str(MODELS_DIR / "match_winner.ubj"))
        logger.info("Saved match_winner.ubj (XGBoost native format)")

    return model, metrics


# ---------------------------------------------------------------------------
# 2. First Innings Score Regressor
# ---------------------------------------------------------------------------
# NOTE: build_score_features() has been moved to features.py where all
# feature-engineering functions live.  Import it from there:
#   from src.features import build_score_features


def train_score_predictor(
    feature_df: pd.DataFrame,
    target_col: str = "final_score",
    save: bool = True,
) -> Tuple[Any, MetricsDict]:
    """
    LightGBM regressor predicting first innings final score.

    Parameters
    ----------
    feature_df : output of build_score_features()
    """
    X = feature_df.drop(columns=[target_col])
    y = feature_df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = lgb.LGBMRegressor(
        n_estimators=400,
        learning_rate=0.04,
        max_depth=5,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(period=-1),
        ],
    )

    preds = model.predict(X_test)
    metrics = {
        "mae": round(mean_absolute_error(y_test, preds), 2),
        "r2": round(r2_score(y_test, preds), 4),
        "feature_importances": dict(zip(X.columns, model.feature_importances_)),
    }

    logger.info("Score predictor — MAE: %.2f | R²: %.4f", metrics["mae"], metrics["r2"])

    if save:
        joblib.dump(model, MODELS_DIR / "score_predictor.pkl")

    return model, metrics


# ---------------------------------------------------------------------------
# 3. Live Win Probability
# ---------------------------------------------------------------------------


def train_win_probability(
    feature_df: pd.DataFrame,
    target_col: str = "batting_team_won",
    save: bool = True,
) -> Tuple[Any, MetricsDict]:
    """
    LightGBM classifier for live (ball-by-ball) win probability.

    Parameters
    ----------
    feature_df : output of features.build_win_probability_features()
    """
    drop_cols = (
        [target_col, "match_id"] if "match_id" in feature_df.columns else [target_col]
    )
    X = feature_df.drop(columns=drop_cols)
    y = feature_df[target_col]

    # Temporal split: last 20% of match_ids as test (simulate real-world hold-out).
    # np.sort ensures the split is chronological by match ID — pd.unique() returns
    # values in order of first appearance, which is NOT guaranteed to be sorted.
    if "match_id" in feature_df.columns:
        match_ids = np.sort(feature_df["match_id"].unique())
        n_test = int(len(match_ids) * 0.2)
        test_ids = set(match_ids[-n_test:])
        mask = feature_df["match_id"].isin(test_ids)
        X_train, X_test = X[~mask], X[mask]
        y_train, y_test = y[~mask], y[mask]
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

    model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=6,
        num_leaves=40,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(period=-1),
        ],
    )

    proba = model.predict_proba(X_test)[:, 1]
    preds = model.predict(X_test)

    metrics = {
        "accuracy": round(accuracy_score(y_test, preds), 4),
        "roc_auc": round(roc_auc_score(y_test, proba), 4),
        "log_loss": round(log_loss(y_test, proba), 4),
        "feature_importances": dict(zip(X.columns, model.feature_importances_)),
    }

    logger.info(
        "Win probability model — AUC: %.4f | Acc: %.4f",
        metrics["roc_auc"],
        metrics["accuracy"],
    )

    if save:
        joblib.dump(model, MODELS_DIR / "win_probability.pkl")

    return model, metrics


def predict_win_curve(
    model: Any,
    match_id: int,
    feature_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Returns over-by-over win probability for the batting team in a given match.

    Parameters
    ----------
    model      : trained win probability model
    match_id   : the IPL match_id to analyse
    feature_df : the same feature_df used for training (must contain match_id col)

    Returns
    -------
    DataFrame with columns: over, win_probability
    """
    match_data = feature_df[feature_df["match_id"] == match_id].copy()
    if match_data.empty:
        raise ValueError(f"match_id {match_id} not found in feature_df")

    drop_cols = ["batting_team_won", "match_id"]
    X = match_data.drop(columns=[c for c in drop_cols if c in match_data.columns])
    proba = model.predict_proba(X)[:, 1]

    return pd.DataFrame(
        {
            "over": match_data["over"].values,
            "win_probability": (proba * 100).round(1),
        }
    )


# ---------------------------------------------------------------------------
# 4. Player of the Match Classifier
# ---------------------------------------------------------------------------
# NOTE: build_potm_features() has been moved to features.py.
# Import it from there:
#   from src.features import build_potm_features


def train_potm_classifier(
    feature_df: pd.DataFrame,
    target_col: str = "is_potm",
    save: bool = True,
) -> Tuple[Any, MetricsDict]:
    """
    XGBoost classifier for Player of the Match prediction.
    Note: class is heavily imbalanced — uses scale_pos_weight.
    """
    X = feature_df.drop(columns=[target_col])
    y = feature_df[target_col]

    neg, pos = (y == 0).sum(), (y == 1).sum()
    spw = neg / pos  # scale_pos_weight for imbalance

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        scale_pos_weight=spw,
        eval_metric="aucpr",
        random_state=42,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    proba = model.predict_proba(X_test)[:, 1]
    preds = model.predict(X_test)

    metrics = {
        "accuracy": round(accuracy_score(y_test, preds), 4),
        "roc_auc": round(roc_auc_score(y_test, proba), 4),
        "avg_precision": round(average_precision_score(y_test, proba), 4),  # PR-AUC
        "precision_pos": round(precision_score(y_test, preds, zero_division=0), 4),
        "recall_pos": round(recall_score(y_test, preds), 4),
        "f1_pos": round(f1_score(y_test, preds, zero_division=0), 4),
        "feature_importances": dict(zip(X.columns, model.feature_importances_)),
        "class_balance": {"negative": int(neg), "positive": int(pos)},
    }

    logger.info("POTM classifier — AUC: %.4f", metrics["roc_auc"])

    if save:
        model.save_model(str(MODELS_DIR / "potm_classifier.ubj"))
        logger.info("Saved potm_classifier.ubj (XGBoost native format)")

    return model, metrics


# ---------------------------------------------------------------------------
# Utility: load saved model
# ---------------------------------------------------------------------------


def load_model(name: str) -> Any:
    """
    Load a saved model by name (without extension).

    Resolution order:
      1. <name>.ubj  — XGBoost native binary JSON (match_winner, potm_classifier)
      2. <name>.pkl  — joblib pickle (score_predictor, win_probability)

    XGBoost models are saved as .ubj to avoid the version-mismatch UserWarning
    that occurs when loading joblib-pickled XGBoost models across versions.
    """
    ubj_path = MODELS_DIR / f"{name}.ubj"
    pkl_path = MODELS_DIR / f"{name}.pkl"

    if ubj_path.exists():
        import xgboost as xgb

        model = xgb.XGBClassifier()
        model.load_model(str(ubj_path))
        return model

    if pkl_path.exists():
        return joblib.load(pkl_path)

    raise FileNotFoundError(
        f"No saved model at {ubj_path} or {pkl_path}. Train it first."
    )
