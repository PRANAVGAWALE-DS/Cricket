"""
05_fix_sklearn_warning.py
-------------------------
Retrains ONLY the win_probability model under the current sklearn version,
eliminating the InconsistentVersionWarning that fires every time uvicorn
starts because win_probability.pkl was serialized under sklearn 1.4.0.

Does NOT touch:
  - match_winner.ubj      (v3, 20 features — leave as-is)
  - potm_classifier.ubj   (leave as-is)
  - score_predictor.pkl   (LightGBM pkl is version-stable — leave as-is)

Run from the project root:
    python notebooks/05_fix_sklearn_warning.py

Output:
    models/win_probability.pkl   (overwritten, now serialized under current sklearn)

Restart uvicorn after running — the LabelEncoder warning will not appear.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import sklearn
import pandas as pd

from src.models import train_win_probability

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("fix_warning")

PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"


def main() -> None:
    logger.info("sklearn version: %s", sklearn.__version__)
    logger.info("Loading win_prob_features.parquet…")

    t0 = time.perf_counter()
    win_prob_feats = pd.read_parquet(PROCESSED / "win_prob_features.parquet")
    logger.info(
        "Loaded %d over-snapshots from %d matches",
        len(win_prob_feats),
        win_prob_feats["match_id"].nunique(),
    )

    logger.info("Retraining win_probability model…")
    _, metrics = train_win_probability(win_prob_feats, save=True)

    logger.info("─" * 55)
    logger.info("RESULTS")
    logger.info("─" * 55)
    logger.info("  AUC (test):  %.4f", metrics["roc_auc"])
    logger.info("  Accuracy:    %.4f", metrics["accuracy"])
    logger.info("  Log-loss:    %.4f", metrics["log_loss"])
    logger.info("  Saved →      %s", MODELS / "win_probability.pkl")
    logger.info("─" * 55)
    logger.info("Done in %.1f s", time.perf_counter() - t0)
    logger.info("Restart uvicorn — the LabelEncoder warning will not appear.")


if __name__ == "__main__":
    main()
