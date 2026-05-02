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
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
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
        use_label_encoder=False,
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
        joblib.dump(model, MODELS_DIR / "match_winner.pkl")

    return model, metrics


# ---------------------------------------------------------------------------
# 2. First Innings Score Regressor
# ---------------------------------------------------------------------------


def build_score_features(
    deliveries: pd.DataFrame, matches: pd.DataFrame
) -> pd.DataFrame:
    """
    Builds over-10 (halfway point) snapshot features to predict final 1st innings score.

    Prediction point: end of over 10 — teams make this calculation at the drinks break.
    Shifting from over-6 to over-10 gives the model 2/3 of the innings as context,
    substantially reducing irreducible uncertainty.

    Features:
        runs_10, wickets_10, current_rr, projected_score,
        scoring_pressure, boundaries_10,
        batting_team_enc, venue_enc, season
    Target: final_score
    """
    is_first = (deliveries["inning"] == 1) & (~deliveries["is_super_over"].astype(bool))

    half = (
        deliveries[is_first & (deliveries["over"] <= 10)]
        .groupby("match_id")
        .agg(
            runs_10=("total_runs", "sum"),
            wickets_10=("is_wicket", "sum"),
            boundaries_10=("batsman_runs", lambda x: ((x == 4) | (x == 6)).sum()),
            batting_team=("batting_team", "first"),
        )
        .reset_index()
    )
    half["current_rr"] = (half["runs_10"] / 10).round(3)
    half["projected_score"] = (half["current_rr"] * 20).round(1)

    full = (
        deliveries[is_first]
        .groupby("match_id")["total_runs"]
        .sum()
        .reset_index(name="final_score")
    )

    df = (
        half.merge(full, on="match_id")
        .merge(
            matches[["id", "venue", "season", "date"]].rename(
                columns={"id": "match_id"}
            ),
            on="match_id",
            how="left",
        )
        .sort_values("date")
        .reset_index(drop=True)
    )

    # Rolling venue avg run rate — no leakage, uses only prior matches
    venue_rr: dict = {}
    venue_rr_col = []
    for _, row in df.iterrows():
        v = row["venue"]
        prior = venue_rr.get(v, [])
        avg = round(sum(prior) / len(prior), 3) if prior else np.nan
        venue_rr_col.append(avg)
        prior.append(row["current_rr"])
        venue_rr[v] = prior

    global_rr = df["current_rr"].mean()
    df["venue_avg_rr"] = pd.Series(venue_rr_col).fillna(global_rr).values
    df["scoring_pressure"] = (df["current_rr"] - df["venue_avg_rr"]).round(3)

    df["batting_team_enc"] = df["batting_team"].astype("category").cat.codes
    df["venue_enc"] = df["venue"].astype("category").cat.codes

    features = [
        "runs_10",
        "wickets_10",
        "current_rr",
        "projected_score",
        "scoring_pressure",
        "boundaries_10",
        "batting_team_enc",
        "venue_enc",
        "season",
        "final_score",
    ]
    return df[features].dropna()


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

    # Temporal split: last 20% of match_ids as test (simulate real-world hold-out)
    if "match_id" in feature_df.columns:
        match_ids = feature_df["match_id"].unique()
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


def build_potm_features(
    deliveries: pd.DataFrame, matches: pd.DataFrame
) -> pd.DataFrame:
    """
    Build player-level per-match features for POTM classification.

    For each (match_id, player) pair:
      - runs_scored, balls_faced, strike_rate
      - wickets_taken, runs_given, economy
      - player_won_team (1 if player's team won)
    Target: is_potm
    """
    batting = (
        deliveries[~deliveries["is_super_over"]]
        .groupby(["match_id", "batsman"])
        .agg(
            runs_scored=("batsman_runs", "sum"),
            balls_faced=("is_legal_delivery", "sum"),
        )
        .reset_index()
        .rename(columns={"batsman": "player"})
    )

    bowling = (
        deliveries[~deliveries["is_super_over"] & deliveries["is_legal_delivery"]]
        .groupby(["match_id", "bowler"])
        .agg(
            wickets_taken=("is_wicket", "sum"),
            runs_given=("total_runs", "sum"),
            balls_bowled=("is_legal_delivery", "sum"),
        )
        .reset_index()
        .rename(columns={"bowler": "player"})
    )

    # Full player list per match
    all_players = pd.concat(
        [
            batting[["match_id", "player"]],
            bowling[["match_id", "player"]],
        ]
    ).drop_duplicates()

    df = all_players.merge(batting, on=["match_id", "player"], how="left")
    df = df.merge(bowling, on=["match_id", "player"], how="left")
    df = df.fillna(0)

    df["strike_rate"] = (
        df["runs_scored"] / df["balls_faced"].replace(0, np.nan) * 100
    ).fillna(0)
    df["economy"] = (
        df["runs_given"] / (df["balls_bowled"] / 6).replace(0, np.nan)
    ).fillna(0)

    # POTM ground truth
    potm = matches[["id", "player_of_match", "winner"]].rename(
        columns={"id": "match_id", "player_of_match": "potm"}
    )
    df = df.merge(potm, on="match_id", how="left")
    df["is_potm"] = (df["player"] == df["potm"]).astype(int)

    # Was player on the winning team?
    bat_team = (
        deliveries[~deliveries["is_super_over"]]
        .groupby(["match_id", "batsman"])["batting_team"]
        .first()
        .reset_index()
        .rename(columns={"batsman": "player"})
    )
    df = df.merge(bat_team, on=["match_id", "player"], how="left")
    df["player_won"] = (df["batting_team"] == df["winner"]).astype(int)

    features = [
        "runs_scored",
        "balls_faced",
        "strike_rate",
        "wickets_taken",
        "runs_given",
        "economy",
        "player_won",
    ]
    return df[features + ["is_potm"]].dropna()


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
        use_label_encoder=False,
        eval_metric="aucpr",
        random_state=42,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    proba = model.predict_proba(X_test)[:, 1]
    preds = model.predict(X_test)

    from sklearn.metrics import (
        precision_score,
        recall_score,
        f1_score,
        average_precision_score,
    )

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
        joblib.dump(model, MODELS_DIR / "potm_classifier.pkl")

    return model, metrics


# ---------------------------------------------------------------------------
# Utility: load saved model
# ---------------------------------------------------------------------------


def load_model(name: str) -> Any:
    """Load a saved model by name (without .pkl extension)."""
    path = MODELS_DIR / f"{name}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"No saved model at {path}. Train it first.")
    return joblib.load(path)
