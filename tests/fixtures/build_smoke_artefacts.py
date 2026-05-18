"""
tests/fixtures/build_smoke_artefacts.py
---------------------------------------
Generates minimal synthetic data and trains lightweight smoke models
so that CI can run the full test suite without the real 150k-row
deliveries dataset or the production .pkl / .ubj / .pt files
(all of which are gitignored).

Run directly:
    python tests/fixtures/build_smoke_artefacts.py

Outputs (written to standard project paths):
    data/raw/matches.csv
    data/raw/deliveries.csv
    data/processed/*.parquet
    models/match_winner.ubj
    models/score_predictor.pkl
    models/win_probability.pkl
    models/potm_classifier.ubj
    models/gru_score_predictor.pt

All models are trained on ~80 synthetic matches — enough for the
validation gate assertions but not representative of real performance.

NOTE: Run `pip install -r requirements.txt -r requirements_tier3.txt`
      before executing this script.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
MODELS = ROOT / "models"

for d in [DATA_RAW, DATA_PROC, MODELS]:
    d.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(42)
N_MATCHES = 80
TEAMS = ["MI", "CSK", "KKR", "RCB", "DC", "SRH", "RR", "KXIP"]
VENUES = [
    "Wankhede Stadium",
    "Eden Gardens",
    "Chinnaswamy Stadium",
    "Arun Jaitley Stadium",
    "Rajiv Gandhi Intl. Stadium",
]
SEASONS = list(range(2012, 2020))


# ---------------------------------------------------------------------------
# 1. Synthetic matches.csv
# ---------------------------------------------------------------------------
print("Generating synthetic matches.csv…")

match_rows = []
for i in range(1, N_MATCHES + 1):
    t1, t2 = RNG.choice(TEAMS, size=2, replace=False)
    toss_w = RNG.choice([t1, t2])
    toss_d = RNG.choice(["bat", "field"])
    winner = RNG.choice([t1, t2])
    season = int(RNG.choice(SEASONS))
    venue = RNG.choice(VENUES)
    win_runs = int(RNG.integers(1, 50)) if winner == t1 else 0
    win_wkts = int(RNG.integers(1, 10)) if winner == t2 else 0
    match_rows.append(
        {
            "id": i,
            "season": season,
            "city": venue.split(",")[0].strip(),
            "date": f"{season}-04-{RNG.integers(1,30):02d}",
            "team1": t1,
            "team2": t2,
            "toss_winner": toss_w,
            "toss_decision": toss_d,
            "result": "normal",
            "dl_applied": 0,
            "winner": winner,
            "win_by_runs": win_runs,
            "win_by_wickets": win_wkts,
            "player_of_match": "Synthetic Player",
            "venue": venue,
            "umpire1": "U1",
            "umpire2": "U2",
        }
    )

matches_df = pd.DataFrame(match_rows)
matches_df.to_csv(DATA_RAW / "matches.csv", index=False)
print(f"  matches.csv: {len(matches_df)} rows")


# ---------------------------------------------------------------------------
# 2. Synthetic deliveries.csv
# ---------------------------------------------------------------------------
print("Generating synthetic deliveries.csv…")

delivery_rows = []
for _, m in matches_df.iterrows():
    for inning in [1, 2]:
        bat = m["team1"] if inning == 1 else m["team2"]
        bowl = m["team2"] if inning == 1 else m["team1"]
        players_bat = [f"{bat}_P{j}" for j in range(1, 12)]
        players_bowl = [f"{bowl}_P{j}" for j in range(1, 12)]
        wickets = 0
        for over in range(1, 21):
            for ball in range(1, 7):
                batter = players_bat[min(wickets, 10)]
                bowler = players_bowl[over % 5]
                bat_runs = int(
                    RNG.choice([0, 1, 2, 4, 6], p=[0.4, 0.3, 0.1, 0.15, 0.05])
                )
                is_wicket = (wickets < 10) and (RNG.random() < 0.04)
                dismissed = batter if is_wicket else np.nan
                if is_wicket:
                    wickets += 1
                delivery_rows.append(
                    {
                        "match_id": m["id"],
                        "inning": inning,
                        "batting_team": bat,
                        "bowling_team": bowl,
                        "over": over,
                        "ball": ball,
                        "batsman": batter,
                        "non_striker": players_bat[min(wickets + 1, 10)],
                        "bowler": bowler,
                        "is_super_over": 0,
                        "wide_runs": 0,
                        "bye_runs": 0,
                        "legbye_runs": 0,
                        "noball_runs": 0,
                        "penalty_runs": 0,
                        "batsman_runs": bat_runs,
                        "extra_runs": 0,
                        "total_runs": bat_runs,
                        "player_dismissed": dismissed,
                        "dismissal_kind": "caught" if is_wicket else np.nan,
                        "fielder": np.nan,
                    }
                )

deliveries_df = pd.DataFrame(delivery_rows)
deliveries_df.to_csv(DATA_RAW / "deliveries.csv", index=False)
print(f"  deliveries.csv: {len(deliveries_df)} rows")


# ---------------------------------------------------------------------------
# 3. Build processed parquets
# ---------------------------------------------------------------------------
print("Building processed parquets…")

from src.data_loader import load_both, save_processed
from src.features import (
    build_match_features_v2,
    build_score_features,
    build_win_probability_features,
    build_potm_features,
)
from src.rolling_features import build_match_features_v3

matches, deliveries = load_both()
save_processed(matches, "matches")
save_processed(deliveries, "deliveries")

mf = build_match_features_v2(matches)
save_processed(mf, "match_features")

mf3 = build_match_features_v3(matches, deliveries)
save_processed(mf3, "match_features_v3")

sf = build_score_features(deliveries, matches)
save_processed(sf, "score_features")

wpf = build_win_probability_features(deliveries, matches)
save_processed(wpf, "win_prob_features")

pf = build_potm_features(deliveries, matches)
save_processed(pf, "potm_features")

print("  All parquets written.")


# ---------------------------------------------------------------------------
# 4. Train smoke models
# ---------------------------------------------------------------------------
print("Training smoke models…")

from src.models import (
    train_match_winner,
    train_score_predictor,
    train_win_probability,
    train_potm_classifier,
)

train_match_winner(mf3, save=True)
print("  match_winner.ubj saved")

train_score_predictor(sf, save=True)
print("  score_predictor.pkl saved")

train_win_probability(wpf, save=True)
print("  win_probability.pkl saved")

# POTM: synthetic data often produces single-class test splits (all not-POTM).
# Wrap in try/except so the fixture builder doesn't abort — the model is still
# saved correctly; roc_auc just can't be computed on a single-class split.
try:
    train_potm_classifier(pf, save=True)
    print("  potm_classifier.ubj saved")
except Exception as exc:
    print(
        f"  potm_classifier warning during eval (single-class split on synthetic data): {exc}"
    )
    # Train without eval to still produce the artefact
    import xgboost as xgb
    from sklearn.model_selection import train_test_split as tts

    _X = pf.drop(columns=["is_potm"])
    _y = pf["is_potm"]
    _neg, _pos = (_y == 0).sum(), (_y == 1).sum()
    _spw = float(_neg) / max(float(_pos), 1)
    _Xtr, _Xte, _ytr, _ = tts(_X, _y, test_size=0.2, random_state=42, stratify=_y)
    _m = xgb.XGBClassifier(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.1,
        scale_pos_weight=_spw,
        eval_metric="logloss",
        random_state=42,
    )
    _m.fit(_Xtr, _ytr, verbose=False)
    _m.save_model(str(MODELS / "potm_classifier.ubj"))
    print("  potm_classifier.ubj saved (fallback path)")

# GRU smoke model
from src.gru_score_model import (
    build_over_sequences,
    build_enc_maps,
    fit_normaliser,
    OverSequenceDataset,
    train_gru,
    save_gru,
    STEP_FEATURES,
)
import torch

over_df = build_over_sequences(deliveries, matches)
enc_maps = build_enc_maps(over_df)
season_range = (float(over_df["season"].min()), float(over_df["season"].max()))
norm_stats = fit_normaliser(over_df, STEP_FEATURES)
device = torch.device("cpu")

ds = OverSequenceDataset(over_df, enc_maps, norm_stats, season_range)
# For CI: tiny split — 80% train, 20% val by index
n_val = max(1, len(ds) // 5)
n_train = len(ds) - n_val
train_ds, val_ds = torch.utils.data.random_split(
    ds, [n_train, n_val], generator=torch.Generator().manual_seed(42)
)

gru_model, gru_metrics = train_gru(
    train_ds,
    val_ds,
    hidden_dim=32,
    num_layers=1,
    dropout=0.0,
    lr=1e-2,
    epochs=10,
    batch_size=16,
    patience=5,
    device=device,
)
save_gru(
    gru_model,
    norm_stats,
    enc_maps,
    season_range,
    gru_metrics,
    str(MODELS / "gru_score_predictor.pt"),
)
print("  gru_score_predictor.pt saved")

print("\nSmoke artefacts ready. Run: pytest tests/")
