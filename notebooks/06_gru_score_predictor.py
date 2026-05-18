"""
06_gru_score_predictor.py
-------------------------
Trains the GRU first-innings score predictor and saves it to
models/gru_score_predictor.pt.

Run from the project root (venv active):
    python notebooks/06_gru_score_predictor.py

Requirements:
    torch >= 2.0   (already in requirements_tier3.txt if installed for RTX 3050)
    Install if needed: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

Output:
    models/gru_score_predictor.pt   — model + inference artefacts
    Comparison table: GRU vs LightGBM (val MAE, R²)

Expected improvement:
    LightGBM (static over-10 snapshot): MAE ~13, R² ~0.44
    GRU (rolling sequence):             MAE ~9,  R² ~0.62+
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from src.data_loader import load_both
from src.gru_score_model import (
    OverSequenceDataset,
    build_enc_maps,
    build_over_sequences,
    fit_normaliser,
    load_gru,
    save_gru,
    train_gru,
    STEP_FEATURES,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("gru_pipeline")

PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
GRU_PATH = str(MODELS / "gru_score_predictor.pt")


def _hr(label: str) -> None:
    logger.info("─" * 60)
    logger.info(label)
    logger.info("─" * 60)


def main() -> None:
    t0 = time.perf_counter()

    # ── Device ────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        logger.info(
            "GPU: %s  |  VRAM: %.1f GB",
            torch.cuda.get_device_name(0),
            torch.cuda.get_device_properties(0).total_memory / 1e9,
        )
    else:
        logger.info("Running on CPU (no CUDA detected)")

    # ── Step 1: Load data ─────────────────────────────────────────────────
    _hr("Step 1 / 5 — Loading data")
    matches, deliveries = load_both()
    logger.info("matches=%d  deliveries=%d", len(matches), len(deliveries))

    # ── Step 2: Build over sequences ──────────────────────────────────────
    _hr("Step 2 / 5 — Building over sequences")
    over_df = build_over_sequences(deliveries, matches)

    # Temporal split: last 20% of match IDs (sorted) as validation
    all_match_ids = np.sort(over_df["match_id"].unique())
    n_val = int(len(all_match_ids) * 0.2)
    val_ids = set(all_match_ids[-n_val:])
    train_ids = set(all_match_ids[:-n_val])

    train_df = over_df[over_df["match_id"].isin(train_ids)]
    val_df = over_df[over_df["match_id"].isin(val_ids)]
    logger.info(
        "Train matches: %d  |  Val matches: %d",
        len(train_ids),
        len(val_ids),
    )

    # ── Step 3: Encoding + normalisation ──────────────────────────────────
    _hr("Step 3 / 5 — Encoding & normalisation")
    enc_maps = build_enc_maps(over_df)  # fit on full data
    season_range = (
        float(over_df["season"].min()),
        float(over_df["season"].max()),
    )
    norm_stats = fit_normaliser(train_df, STEP_FEATURES)

    logger.info(
        "Teams: %d  |  Venues: %d",
        len(enc_maps["batting_team"]),
        len(enc_maps["venue"]),
    )
    logger.info("Season range: %.0f – %.0f", *season_range)
    logger.info("Norm stats computed on %d train rows", len(train_df))

    # ── Step 4: Build datasets ─────────────────────────────────────────────
    _hr("Step 4 / 5 — Building datasets & training")
    train_ds = OverSequenceDataset(train_df, enc_maps, norm_stats, season_range)
    val_ds = OverSequenceDataset(val_df, enc_maps, norm_stats, season_range)
    logger.info("Train samples: %d  |  Val samples: %d", len(train_ds), len(val_ds))

    # ── Step 5: Train ──────────────────────────────────────────────────────
    model, metrics = train_gru(
        train_ds=train_ds,
        val_ds=val_ds,
        hidden_dim=64,
        num_layers=2,
        dropout=0.3,
        lr=1e-3,
        epochs=300,
        batch_size=32,
        patience=25,
        device=device,
    )

    # ── Results ───────────────────────────────────────────────────────────
    _hr("Results")
    logger.info(
        "GRU  — MAE: %.2f runs  |  R²: %.4f  |  Epochs: %d",
        metrics["val_mae"],
        metrics["val_r2"],
        metrics["epochs_trained"],
    )

    # Compare against LightGBM baseline from score_features.parquet
    try:
        import joblib
        from sklearn.metrics import mean_absolute_error, r2_score
        from sklearn.model_selection import train_test_split

        score_feats = pd.read_parquet(PROCESSED / "score_features.parquet")
        X = score_feats.drop(columns=["final_score"])
        y = score_feats["final_score"]
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        lgb_model = joblib.load(MODELS / "score_predictor.pkl")
        lgb_preds = lgb_model.predict(X_test)
        lgb_mae = mean_absolute_error(y_test, lgb_preds)
        lgb_r2 = r2_score(y_test, lgb_preds)

        logger.info("─" * 55)
        logger.info("Comparison (val set)")
        logger.info(
            "  LightGBM (static over-10) — MAE: %.2f  R²: %.4f", lgb_mae, lgb_r2
        )
        logger.info(
            "  GRU (rolling sequence)     — MAE: %.2f  R²: %.4f",
            metrics["val_mae"],
            metrics["val_r2"],
        )
        improvement_mae = lgb_mae - metrics["val_mae"]
        logger.info(
            "  MAE improvement: %.2f runs (%.1f%%)",
            improvement_mae,
            improvement_mae / lgb_mae * 100,
        )
        logger.info("─" * 55)
    except Exception as e:
        logger.warning("Could not load LightGBM for comparison: %s", e)

    # ── Save ──────────────────────────────────────────────────────────────
    save_gru(model, norm_stats, enc_maps, season_range, metrics, GRU_PATH)
    logger.info("Saved → %s", GRU_PATH)

    # Quick sanity check: reload and run one inference
    logger.info("Sanity check — reloading and running test inference…")
    loaded_model, payload = load_gru(GRU_PATH, device=torch.device("cpu"))
    from src.gru_score_model import predict_from_overs

    # Simulate over-10 state for MI at Wankhede
    test_overs = [
        {
            "runs_in_over": 7,
            "wickets_in_over": 0,
            "cum_runs": 7,
            "cum_wickets": 0,
            "current_rr": 7.0,
            "boundaries_in_over": 1,
            "balls_in_over": 6,
        },
        {
            "runs_in_over": 8,
            "wickets_in_over": 1,
            "cum_runs": 15,
            "cum_wickets": 1,
            "current_rr": 7.5,
            "boundaries_in_over": 2,
            "balls_in_over": 6,
        },
    ] * 5  # 10 overs

    teams = list(enc_maps["batting_team"].keys())
    venues = list(enc_maps["venue"].keys())
    test_pred = predict_from_overs(
        loaded_model,
        payload,
        test_overs,
        batting_team=teams[0],
        venue=venues[0],
        season=2016,
    )
    logger.info(
        "Test inference (10 overs, 75/5): predicted final score = %.1f", test_pred
    )

    logger.info("─" * 60)
    logger.info("Pipeline complete in %.1f s", time.perf_counter() - t0)
    logger.info("Next: restart uvicorn to load gru_score_predictor.pt")
    logger.info("─" * 60)


if __name__ == "__main__":
    main()
