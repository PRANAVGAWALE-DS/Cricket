"""
04_rolling_features.py
----------------------
Pipeline: compute rolling player-form → save parquet → retrain match winner.

Run from the project root:
    python notebooks/04_rolling_features.py

Outputs
-------
data/processed/team_rolling_form.parquet
    One row per (match_id, team). Used by the API at inference time to look
    up the current rolling form of any team.

data/processed/match_features_v3.parquet
    Full match-winner feature matrix (v2 base + 11 rolling features).

models/match_winner.pkl
    Retrained XGBoost classifier. Replaces the previous version.
    Restart uvicorn after running this script.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

# Make src/ importable when running from project root
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_loader import load_both  # noqa: E402
from src.models import train_match_winner  # noqa: E402
from src.rolling_features import (  # noqa: E402
    build_match_features_v3,
    compute_team_rolling_form,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("pipeline")

PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"


def _hr(label: str) -> None:
    logger.info("─" * 60)
    logger.info(label)
    logger.info("─" * 60)


def main() -> None:
    t0 = time.perf_counter()

    # ── 1. Load raw data ──────────────────────────────────────────────────
    _hr("Step 1 / 4 — Loading data")
    matches, deliveries = load_both()
    logger.info("matches: %d rows | deliveries: %d rows", len(matches), len(deliveries))

    # ── 2. Compute team rolling form ──────────────────────────────────────
    _hr("Step 2 / 4 — Computing team rolling form (n=5)")
    team_form = compute_team_rolling_form(deliveries, matches, n_matches=5)

    out_path = PROCESSED / "team_rolling_form.parquet"
    team_form.to_parquet(out_path, index=False)
    logger.info(
        "Saved %s — %d rows, %d teams",
        out_path,
        len(team_form),
        team_form["team"].nunique(),
    )

    # Quick sanity: print last rolling form per team so you can verify
    logger.info("Latest rolling form per team:")
    latest_idx = team_form.groupby("team")["match_id"].idxmax()
    latest = team_form.loc[latest_idx].sort_values("team")
    for _, row in latest.iterrows():
        logger.info(
            "  %-6s  bat_avg=%.1f  bat_sr=%.1f  bowl_econ=%.2f  bowl_sr=%.1f",
            row["team"],
            row["rolling_bat_avg"],
            row["rolling_bat_sr"],
            row["rolling_bowl_econ"],
            row["rolling_bowl_sr"],
        )

    # ── 3. Build enhanced match features ─────────────────────────────────
    _hr("Step 3 / 4 — Building match_features_v3")
    feature_df = build_match_features_v3(matches, deliveries, n_matches=5)

    feat_path = PROCESSED / "match_features_v3.parquet"
    feature_df.to_parquet(feat_path, index=False)
    n_feat = len(feature_df.columns) - 1  # exclude target
    logger.info(
        "Saved %s — %d rows, %d features",
        feat_path,
        len(feature_df),
        n_feat,
    )
    logger.info(
        "Feature columns: %s", list(feature_df.drop(columns=["team1_won"]).columns)
    )

    # Class balance check
    pos = int(feature_df["team1_won"].sum())
    neg = len(feature_df) - pos
    logger.info("Class balance — team1_won=1: %d | team1_won=0: %d", pos, neg)

    # ── 4. Retrain match winner ───────────────────────────────────────────
    _hr("Step 4 / 4 — Retraining match winner model")
    logger.info("Training XGBoost on %d rows × %d features…", len(feature_df), n_feat)

    model, metrics = train_match_winner(feature_df, target_col="team1_won", save=True)

    logger.info("─" * 60)
    logger.info("RESULTS")
    logger.info("─" * 60)
    logger.info("  AUC (test):    %.4f", metrics["roc_auc"])
    logger.info("  Accuracy:      %.4f", metrics["accuracy"])
    logger.info("  Log-loss:      %.4f", metrics["log_loss"])
    logger.info(
        "  CV AUC:        %.4f ± %.4f (5-fold)",
        metrics["cv_roc_auc_mean"],
        metrics["cv_roc_auc_std"],
    )
    logger.info("  Model saved →  %s", MODELS / "match_winner.pkl")

    # Feature importance top-10
    fi = sorted(
        metrics["feature_importances"].items(), key=lambda x: x[1], reverse=True
    )
    logger.info("Top-10 feature importances:")
    for feat, imp in fi[:10]:
        bar = "█" * int(imp * 200)
        logger.info("  %-30s  %.4f  %s", feat, imp, bar)

    elapsed = time.perf_counter() - t0
    logger.info("─" * 60)
    logger.info("Pipeline complete in %.1f s", elapsed)
    logger.info("Next step: restart uvicorn to load the new match_winner.pkl")
    logger.info("─" * 60)


if __name__ == "__main__":
    main()
