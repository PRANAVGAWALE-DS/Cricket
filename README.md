<div align="center">

# 🏏 Cricket ML — IPL Prediction Suite

**Production-grade ML pipeline for IPL analytics**  
*5 models · 7 API endpoints · Live Streamlit dashboard · Dockerised · CI/CD*

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-HuggingFace_Spaces-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/spaces/PG-AIML/Cricket)
[![GitHub Repo](https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github)](https://github.com/PRANAVGAWALE-DS/Cricket)
[![CI](https://img.shields.io/github/actions/workflow/status/PRANAVGAWALE-DS/Cricket/ci.yml?branch=main&style=for-the-badge&label=CI)](https://github.com/PRANAVGAWALE-DS/Cricket/actions)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)](LICENSE)

<br/>

<img src="https://img.shields.io/badge/XGBoost-Match_Winner_AUC_0.797-FF6B35?style=flat-square"/>
<img src="https://img.shields.io/badge/LightGBM-Win_Probability_AUC_0.822-00B4D8?style=flat-square"/>
<img src="https://img.shields.io/badge/PyTorch_GRU-Score_Predictor-EE4C2C?style=flat-square"/>
<img src="https://img.shields.io/badge/Tests-43_passed-22C55E?style=flat-square"/>

> **Live demo →** [Visit Hugging Face Space](https://huggingface.co/spaces/PG-AIML/Cricket)

</div>

---

## 📑 Table of Contents

- [What Is This?](#what-is-this)
- [Model Results](#model-results)
- [Architecture](#architecture)
- [ML Models](#ml-models)
- [Project Structure](#project-structure)
- [Notebooks](#notebooks)
- [Key Findings](#key-findings)
- [Quickstart](#quickstart)
- [API Reference](#api-reference)
- [Dashboard Pages](#dashboard-pages)
- [Development & Testing](#development-testing)
- [Engineering Highlights](#engineering-highlights)
- [Roadmap](#roadmap)
- [Stack](#stack)
- [Dataset](#dataset)

---

## 📖 What Is This?

A complete, end-to-end ML system that turns raw IPL ball-by-ball data into live predictions. Built with production engineering practices — not a notebook project.

**What makes it different:**

- Match Winner AUC jumped **0.530 → 0.797** by engineering rolling player-form features — *without adding a single new data source*
- **GRU sequence model** enables score prediction at *any* over (1–20), not just over 10
- **Real-time Inference:** Derived features (cumulative runs/wickets) are calculated server-side in the API, minimizing client payload.
- **Leak-Proof Training:** All rolling features use a `shift(1)` guard to ensure the model never sees the outcome of the current match.
- **Production-Grade Serving:** 7-endpoint FastAPI backend + 5-page Streamlit dashboard deployed on HF Spaces.
- **43 automated tests** across 7 test classes with GitHub Actions CI on every push

---

## 📊 Model Results

| Model | Algorithm | Metric | Value | Notes |
|---|---|---|---|---|
| **Match Winner** | XGBoost | AUC | **0.797** | +0.267 from rolling features |
| **Match Winner** | XGBoost | Accuracy | **0.748** | CV AUC: 0.800 ± 0.045 |
| **Score Predictor** | LightGBM | MAE | **18 runs** | Static over-10 snapshot |
| **Score Predictor** | LightGBM | R² | **0.52** | — |
| **GRU Score Predictor** | PyTorch GRU | Val MAE | **18 runs** | Any over 1–20 (temporal val split) |
| **GRU Score Predictor** | PyTorch GRU | Val R² | **0.33** | Val split — see note ¹ |
| **Win Probability** | LightGBM | AUC | **0.822** | Ball-by-ball live |
| **Win Probability** | LightGBM | Accuracy | **0.712** | — |
| **POTM Classifier** | XGBoost | AUC | **0.972** | ROC-AUC — see note ² |

> ¹ **GRU split:** Uses a temporal hold-out (last 20% of match IDs). These are *validation* metrics — the GRU has no separate held-out test set independent of the validation set used for early stopping.
>
> ² **POTM AUC caveat:** With a 1:20 class imbalance, ROC-AUC 0.972 is inflated; a naive "never POTM" classifier achieves ~95% accuracy. **PR-AUC 0.67 and Recall 0.927 are the primary evaluation metrics.**

### Match Winner AUC Journey

```
v1  base features (9)          ████░░░░░░░░░░░░░░░░  0.530
v3  rolling form  (20)         ████████████████░░░░  0.797  ▲ +50% relative
```

> **Key insight:** The entire AUC jump came from feature engineering alone — rolling per-player batting average, strike rate, economy, and bowling SR aggregated to team level with a `shift(1)` leak guard. No new data, no architecture change, no HPO.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Cricket ML Stack                         │
│                                                                   │
│  Raw Data (756 matches · 179k balls)                             │
│  ──────────────────────────────────                              │
│  matches.csv  ──▶  match_features_v3   ──▶  XGBoost (winner)    │
│  deliveries   ──▶  score_features      ──▶  LightGBM (score)    │
│  .csv         ──▶  win_prob_features   ──▶  LightGBM (winprob)  │
│               ──▶  potm_features       ──▶  XGBoost (potm)      │
│               ──▶  over_sequences      ──▶  PyTorch GRU          │
│               ──▶  rolling_form        ──▶  (injected at infer.) │
│                                                                   │
│  ┌─────────────────────┐  HTTP  ┌──────────────────────────┐    │
│  │  FastAPI  :8000     │◀──────▶│  Streamlit  :8501/:7860  │    │
│  │  7 endpoints        │        │  5 interactive pages      │    │
│  │  Pydantic v2        │        │  live GRU inference       │    │
│  └─────────────────────┘        └──────────────────────────┘    │
│                                                                   │
│  Docker Compose (local)  ·  HF Spaces Docker (production)       │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🧠 ML Models

### 1 · Match Winner Classifier

Predicts pre-match win probability for both teams using team-level rolling player form, historical win rates, toss information, venue, and season.

| Property | Detail |
|---|---|
| Algorithm | XGBoost classifier |
| Features (v3) | 20: 9 base (toss, venue, win rates) + 11 rolling player-form |
| Target | `team1_won` (binary) |
| Split | 80% train / 20% test, stratified |
| CV | 5-fold StratifiedKFold, AUC objective |
| **Test AUC** | **0.797** |
| **CV AUC** | **0.800 ± 0.045** |
| **Accuracy** | **0.748** |
| Serialisation | `.ubj` (XGBoost native — no pickle version warnings) |

**Feature importance insight:** `team2_rolling_bat_avg` is the #1 predictor, followed by `rolling_bat_avg_diff`. All top-10 features are rolling player-form columns — historical win rates (`win_rate_diff`) don't even appear in the top 10. The chasing team's batting strength dominates pre-match signal in T20 cricket.

---

### 2 · Score Predictor (LightGBM)

Predicts first-innings final score from a static snapshot at the end of over 10.

| Property | Detail |
|---|---|
| Algorithm | LightGBM regressor |
| Features | runs_10, wickets_10, current_rr, projected_score, scoring_pressure, boundaries_10, batting_team_enc, venue_enc, season |
| Target | `final_score` (runs) |
| Split | 80% train / 20% test |
| Early stopping | 50 rounds on validation MAE |
| **Test MAE** | **18 runs** |
| **Test R²** | **0.52** |
| Serialisation | `.pkl` (joblib) |

> **Derived features:** `projected_score = current_rr × 20` (naive linear projection, also returned as `projected_naive` in the API response). `scoring_pressure` is defined in `src/features.py` — see that file for the exact formula.

**Limitation:** Single snapshot at over 10 only — does not update as innings progresses. See GRU model below for over-by-over inference.

---

### 3 · GRU Score Predictor

2-layer PyTorch GRU that ingests each over as a timestep, enabling score prediction after any number of completed overs (1–20).

| Property | Detail |
|---|---|
| Algorithm | 2-layer GRU (hidden=64, dropout=0.3) |
| Input per step | 10 features: 7 dynamic (runs, wickets, RR, boundaries, cumulative) + 3 static context (team, venue, season) |
| Sequence handling | `pack_padded_sequence` for variable-length innings |
| Split | Temporal — last 20% of match IDs as validation |
| Train / Val samples | 11,966 / 2,986 (one per prefix length per match) |
| Early stopping | patience=25 on val MSE |
| **Val MAE** | **18 runs** |
| **Val R²** | **0.33** |
| Serialisation | `.pt` (torch.save — includes norm_stats, enc_maps, meta) |

**On GRU vs LightGBM:** The GRU's architectural advantage is over-by-over inference — it accepts live ball data as it arrives, while LightGBM requires a complete over-10 snapshot. MAE is tied at 18 runs on this dataset, but GRU R² (0.33) lags LightGBM R² (0.52) — the gap is expected to close as training data grows. Adding Cricsheet data (~5,000 matches) is the primary lever for improving GRU accuracy.

---

### 4 · Win Probability Model

Ball-by-ball live win probability for the chasing team in the 2nd innings.

| Property | Detail |
|---|---|
| Algorithm | LightGBM classifier |
| Features | over, runs_scored, wickets_fallen, current_rr, required_rr, balls_remaining, runs_required, venue_enc, batting_team_enc |
| Target | `batting_team_won` (binary) |
| Split | Temporal — last 20% of match IDs as test |
| Early stopping | 50 rounds |
| **Test AUC** | **0.822** |
| **Test Accuracy** | **0.712** |
| Serialisation | `.pkl` (joblib, retrained under sklearn 1.7.2) |

---

### 5 · POTM Classifier

Predicts Player of the Match probability for each player given their match performance.

| Property | Detail |
|---|---|
| Algorithm | XGBoost classifier |
| Features | runs_scored, balls_faced, wickets_taken, runs_given, balls_bowled, batting_avg, strike_rate, economy, boundaries (per-player match totals) |
| Target | `is_potm` (binary, heavily imbalanced — 1:20 ratio) |
| Imbalance handling | `scale_pos_weight = neg/pos` |
| **Test AUC** | **0.972** |
| **PR-AUC** | **0.67** |
| **Recall (POTM class)** | **0.927** |
| Serialisation | `.ubj` (XGBoost native) |

---

## 🗂️ Project Structure

```
Cricket/
├── 📁 data/                          ← place Kaggle CSVs here (not committed)
│   ├── matches.csv
│   └── deliveries.csv
│
├── 📁 models/                        ← written by make train (not committed)
│   ├── match_winner_v1.ubj
│   ├── score_predictor_v1.pkl
│   ├── win_probability_v1.pkl
│   ├── potm_classifier_v1.ubj
│   └── gru_score_predictor_v1.pt
│
├── 📁 api/
│   ├── main.py              FastAPI — 7 routes, lifespan startup, GRU loader
│   └── schemas.py           Pydantic v2 request/response contracts
│
├── 📁 src/
│   ├── data_loader.py       load_matches(), load_deliveries(), save_processed()
│   ├── features.py          build_*_features() — all feature engineering
│   ├── models.py            train_*() — training + XGBoost .ubj serialisation
│   ├── rolling_features.py  Leak-free rolling player-form (shift(1) guard)
│   └── gru_score_model.py   PyTorch GRU — Dataset, training loop, save/load
│
├── 📁 notebooks/
│   ├── 01_eda.py            EDA — 17 Plotly charts
│   ├── 02_feature_engineering.py
│   ├── 03_modeling.py       Base models (score, win prob, potm)
│   ├── 04_rolling_features.py  Rolling form → retrain match winner
│   ├── 05_fix_sklearn_warning.py  Retrain win_prob under sklearn 1.7.2
│   └── 06_gru_score_predictor.py  GRU training pipeline
│
├── 📁 tests/
│   ├── conftest.py
│   ├── test_cricket_ml.py   43 tests across 7 classes
│   └── fixtures/
│       └── build_smoke_artefacts.py  CI synthetic data + model builder
│
├── 📁 docker/
│   ├── Dockerfile.api
│   └── Dockerfile.streamlit
│
├── 📁 scripts/
│   └── upload_to_hub.py
│
├── 📁 .github/workflows/
│   └── ci.yml               Lint → Tests → Docker build (3 jobs)
│
├── Dockerfile               HF Spaces single-container deployment
├── start.sh                 Starts FastAPI + Streamlit in same container (HF Spaces constraint — not recommended for production; use docker-compose.yml with separate containers instead)
├── streamlit_app.py         5-page dashboard
├── docker-compose.yml
├── Makefile
├── requirements.txt               # core ML + data stack (XGBoost, LightGBM, PyTorch, pandas)
└── requirements_tier3.txt         # + FastAPI, Streamlit, Uvicorn, deployment extras
```

---

## 📓 Notebooks

> Notebooks are plain Python scripts (`.py`), not `.ipynb` files — run with `python notebooks/NN_name.py`. No Jupyter installation required; all outputs (charts, parquets) are written to disk.

### `01_eda.py` — Exploratory Data Analysis

| Section | Detail |
|---|---|
| Data quality | Missing-value audit, team abbreviation normalisation, over-indexing validation |
| Score distributions | First-innings score by venue, season, team |
| Win analysis | Toss impact, venue win rates, home/away patterns |
| Player analysis | Top run-scorers, wicket-takers, economy rates |
| Phase analysis | Powerplay / middle / death over run rates |
| Seasonal trends | Scoring evolution 2008–2019, team performance trajectories |
| Charts | 17 interactive Plotly HTML charts saved to `notebooks/.charts/` |

### `02_feature_engineering.py` — Feature Engineering

| Section | Detail |
|---|---|
| Match features (v2) | 9 base features: toss, venue_enc, team_enc, win rates |
| Score features | Over-10 snapshot: runs, wickets, RR, boundaries, pressure |
| Win prob features | Ball-by-ball: required RR, balls remaining, runs required |
| POTM features | Per-player: batting avg, SR, wickets, economy |
| Encoding | `pd.astype("category").cat.codes` — alphabetical, reproducible |

### `04_rolling_features.py` — Rolling Player Form

| Section | Detail |
|---|---|
| Per-player batting | Runs, balls, dismissals → avg, SR per match |
| Per-player bowling | Wickets (bowler-credited only), runs conceded, economy |
| Leak guard | `shift(1)` before rolling window — current match always excluded |
| Cold-start | Global dataset mean/median fills first-match entries |
| Team aggregation | Mean across all players who appeared for the team |
| Output | 20-feature `match_features_v3.parquet` + `team_rolling_form.parquet` |

### `06_gru_score_predictor.py` — GRU Training

| Section | Detail |
|---|---|
| Sequence building | One row per over → prefix sequences (N match × 20 overs = ~15k samples) |
| Temporal split | Last 20% of match IDs as validation — no data leakage |
| Normalisation | Fit on train only, applied to train + val |
| Training | Adam, ReduceLROnPlateau, gradient clipping (max_norm=5), patience=25 |
| Comparison | GRU vs LightGBM printed on same val set |
| Artefact | `.pt` containing state_dict, norm_stats, enc_maps, season_range, meta |

---

## 🔑 Key Findings

| Metric | Value |
|---|---|
| Top feature for match winner | `team2_rolling_bat_avg` (importance 0.16) |
| AUC improvement from rolling features | +0.267 (+50% relative) |
| Win probability AUC | 0.822 (ball-by-ball, temporal split) |
| POTM recall (POTM class) | 92.7% at 97.2% AUC |
| Score predictor MAE | 18 runs over-10 snapshot |
| Most predictive over for win prob | Overs 15–18 (model confidence highest) |
| Dataset size | 756 matches, 179,078 deliveries, 14 teams, 41 venues |
| Seasons covered | 2008–2019 (IPL 1–12) |

---

## ⚡ Quickstart

```bash
git clone https://github.com/PRANAVGAWALE-DS/Cricket.git
cd Cricket
pip install -r requirements.txt

# CPU torch (default):
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu

# CUDA 12.1 torch (recommended for GPU training):
# pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

**Download the dataset** from [Kaggle — nowke9/ipldata](https://www.kaggle.com/datasets/nowke9/ipldata) and place `matches.csv` and `deliveries.csv` into the `data/` directory:
```
Cricket/
└── data/
    ├── matches.csv
    └── deliveries.csv
```

```bash
# Run full training pipeline (~30 min, GRU is bottleneck)
# Note: notebooks 01 (EDA) and 02 (feature engineering) are one-time exploratory scripts;
# their outputs are pre-committed as parquets. make train runs 03 → 04 → 06 → 05.
make train

# Terminal 1 — FastAPI
# Windows CMD:
set PYTHONPATH=.
# Linux / macOS / Git Bash:  export PYTHONPATH=.
# PowerShell:                $env:PYTHONPATH="."
uvicorn api.main:app --host 0.0.0.0 --port 8000

# Terminal 2 — Streamlit
streamlit run streamlit_app.py
```

| Service | URL |
|---|---|
| FastAPI docs | http://localhost:8000/docs |
| API health | http://localhost:8000/health |
| Dashboard | http://localhost:8501 |

**Docker Compose:**
```bash
docker compose up --build
# API → :8000   Dashboard → :8501
# If either port is already in use, override: API_PORT=8080 DASHBOARD_PORT=8502 docker compose up --build
```

> **Conda users:** An `environment.yml` is not currently tracked. Use `pip install -r requirements.txt` inside your conda env (Python 3.11 recommended).

**Makefile targets:**
```bash
make install       # install all dependencies + torch
make train         # run notebooks 03 → 04 → 06 → 05 (01 & 02 are one-time EDA, outputs pre-committed)
make train-gru     # train GRU only
make api           # start FastAPI
make dashboard     # start Streamlit
make docker-up     # full stack via Docker Compose
make lint          # ruff
```

---

## 🔌 API Reference

**Base URL:** `http://localhost:8000`

<details>
<summary><code>GET /health</code> — API status + loaded models</summary>

```bash
curl http://localhost:8000/health
```
```json
{
  "status": "ok",
  "models_loaded": {
    "match_winner": true, "score_predictor": true,
    "win_probability": true, "potm_classifier": true,
    "gru_score_predictor": true
  },
  "teams": ["CSK", "DC", "KKR", "MI", "..."],
  "venues": ["Eden Gardens", "Wankhede Stadium", "..."]
}
```
</details>

<details>
<summary><code>GET /matches</code> — All match IDs available for win-curve</summary>

```bash
curl http://localhost:8000/matches
```
```json
{ "match_ids": [1, 2, 3, "..."], "count": 754 }
```
> 2 of the 756 raw matches are excluded (no completed 2nd innings — rain/D/L abandonment). Only matches with full 2nd-innings delivery data have a usable win-curve.
</details>

<details>
<summary><code>POST /predict/match-winner</code> — Pre-match win probability</summary>

```bash
curl -X POST http://localhost:8000/predict/match-winner \
  -H "Content-Type: application/json" \
  -d '{"team1":"MI","team2":"CSK","venue":"Wankhede Stadium","toss_winner":"team1","toss_decision":"bat","season":2019}'
```
```json
{
  "team1": "MI", "team2": "CSK",
  "team1_win_probability": 54.3,
  "team2_win_probability": 45.7,
  "model_version": "match_winner_v1"
}
```
`toss_winner` must be `"team1"` or `"team2"`. `toss_decision` must be `"bat"` or `"field"`.
</details>

<details>
<summary><code>POST /predict/score</code> — LightGBM score predictor (over-10 snapshot)</summary>

```bash
curl -X POST http://localhost:8000/predict/score \
  -H "Content-Type: application/json" \
  -d '{"batting_team":"MI","venue":"Wankhede Stadium","season":2019,"runs_10":68,"wickets_10":2,"boundaries_10":9}'
```
```json
{
  "predicted_final_score": 174.5,
  "confidence_interval_low": 161.5,
  "confidence_interval_high": 187.5,
  "current_rr": 6.8,
  "projected_naive": 136.0,
  "model_version": "score_predictor_v1"
}
```
</details>

<details>
<summary><code>POST /predict/score/gru</code> — GRU score predictor (any over)</summary>

```bash
curl -X POST http://localhost:8000/predict/score/gru \
  -H "Content-Type: application/json" \
  -d '{"batting_team":"MI","venue":"Wankhede Stadium","season":2019,"overs":[{"runs_in_over":8,"wickets_in_over":0,"boundaries_in_over":2},{"runs_in_over":6,"wickets_in_over":1,"boundaries_in_over":1}]}'
```
```json
{
  "predicted_final_score": 168.3,
  "confidence_interval_low": 150.3,
  "confidence_interval_high": 186.3,
  "overs_seen": 2,
  "model": "GRU"
}
```
Pass completed overs in chronological order. Cumulative features derived server-side.
</details>

<details>
<summary><code>GET /predict/win-curve/{match_id}</code> — Over-by-over win probability</summary>

```bash
curl http://localhost:8000/predict/win-curve/335982
```
```json
{
  "match_id": 335982, "batting_team": "MI", "bowling_team": "CSK",
  "curve": [
    {"over": 1, "win_probability": 48.2},
    {"over": 2, "win_probability": 51.7}
  ],
  "actual_winner": "MI",
  "model_version": "win_probability_v1"
}
```
</details>

<details>
<summary><code>POST /predict/potm</code> — Player of the Match predictor</summary>

```bash
curl -X POST http://localhost:8000/predict/potm \
  -H "Content-Type: application/json" \
  -d '{"players":[{"player_name":"Rohit Sharma","runs_scored":78,"balls_faced":48,"wickets_taken":0,"runs_given":0,"balls_bowled":0,"player_won":1}]}'
```
```json
{
  "predicted_potm": "Rohit Sharma",
  "players": [
    {"player_name": "Rohit Sharma", "potm_probability": 81.4,
     "strike_rate": 162.5, "economy": 0.0, "rank": 1}
  ],
  "model_version": "potm_classifier_v1"
}
```
</details>

---

## 📱 Dashboard Pages

| Page | What it does |
|---|---|
| 🏏 **Match Winner** | Pre-match probability gauge + team comparison. Uses v3 model with rolling player form. |
| 📊 **Score Predictor** | Two tabs: LightGBM (over-10 snapshot) vs GRU (enter overs one-by-one, live prediction curve updates after each over) |
| 📈 **Win Probability** | Select any historical match → animated over-by-over win probability curve **(2nd innings only — chasing team's live win probability)** |
| 🏆 **POTM Predictor** | Enter player stats → ranked Player of the Match probabilities |
| 🔍 **Player Stats** | Career batting/bowling stats, season trend charts, dual-axis season charts, all-time leaderboards |

---

## 🧪 Development & Testing

```bash
make lint          # ruff check
pytest tests/ -v   # 43 tests
pytest tests/ --cov=src --cov=api --cov-report=term-missing  # with coverage
```

| Module | Tests | What's covered |
|---|---|---|
| `TestDataLoader` | 7 | Schema validation, over indexing, team abbreviations |
| `TestFeatureEngineering` | 7 | Column presence, no nulls, binary targets, positive scores |
| `TestRollingFeatures` | 5 | Parquet exists, schema, no nulls, plausible value ranges |
| `TestModelSerialisation` | 8 | File existence + load + predict round-trip for all 5 models |
| `TestGRUModel` | 3 | Load, predict_from_overs, meta keys present |
| `TestAPIRoutes` | 9 | All 7 endpoints via FastAPI TestClient, 404 on invalid match |
| `TestModelValidationGate` | 4 | AUC/MAE thresholds — fails CI if model degrades |

CI runs on every push: **Lint** → **Tests + model gate** → **Docker build smoke test** (3 parallel jobs). Smoke artefact builder generates 80 synthetic matches so CI never needs real CSVs or production models.

---

## 🧠 Engineering Highlights

**Leak-free rolling features**
`rolling_features.py` applies `shift(1)` before the rolling window — the current match's stats are always excluded from the feature vector. Cold-start entries (player's first match) are filled with the global dataset mean.

**XGBoost native serialisation**
XGBoost models saved as `.ubj` via `model.save_model()` — XGBoost's own binary JSON format. Eliminates the `UserWarning` that fires when loading joblib-pickled XGBoost models across version upgrades.

**GRU with variable-length sequences**
`pack_padded_sequence` handles innings that end early (all-out before over 20). Training uses a temporal split by match ID — no data leakage between train and validation sequences.

**Encoding compatibility without a stored encoder**
Models use `pd.astype("category").cat.codes` (alphabetical ordering). `api/main.py` reconstructs the same maps at startup from the processed parquets — inference encoding always matches training without storing a separate `LabelEncoder` artefact.

> **Scope caveat:** This encoding is safe only for teams and venues present in the 2008–2019 training data. Any team or venue that did not appear in that dataset will receive an out-of-distribution code at inference time. Post-2019 IPL franchises (e.g. Lucknow Super Giants, Gujarat Titans) require data re-ingestion and model retraining before being used as inputs.

**Lifespan startup (FastAPI)**
Migrated from deprecated `@app.on_event("startup")` to `@asynccontextmanager lifespan` — required for correct TestClient startup behaviour in pytest.

---

## 🗺️ Roadmap

> Completed items are listed in build order (not priority order).

- [x] Tier 3 — FastAPI + Streamlit + Docker + Player Stats explorer
- [x] Tier 1 — Rolling player-form features (Match Winner AUC 0.530 → 0.797)
- [x] Tier 2 — GRU sequence score predictor (over-by-over inference)
- [x] GitHub Actions CI (43 tests, 3 jobs)
- [x] HF Spaces Docker deployment
- [ ] Cricsheet data ingestion (IPL 2020–2024, ~5k matches) — GRU target MAE < 15
- [ ] Win Probability GRU (target AUC > 0.90)
- [ ] Fantasy points optimizer (multi-constraint integer program: credit budget, role slots, captain/VC multipliers)

---

## 📦 Stack

| Layer | Tools |
|---|---|
| ML | XGBoost, LightGBM, PyTorch (GRU), scikit-learn |
| Data | Pandas, NumPy, PyArrow |
| API | FastAPI, Pydantic v2, Uvicorn |
| Dashboard | Streamlit, Plotly |
| DevOps | Docker, GitHub Actions, HF Spaces |
| Testing | pytest, pytest-cov, httpx (FastAPI TestClient) |

---

## 📄 Dataset

| Property | Value |
|---|---|
| Source | [Kaggle — nowke9/ipldata](https://www.kaggle.com/datasets/nowke9/ipldata) |
| Matches | 756 |
| Deliveries | 179,078 |
| Seasons | 2008–2019 (IPL 1–12) |
| Teams | 14 (including defunct: DC_old, GL, KTK, PW, RPS) |
| Venues | 41 |

### Known Data Issues

| Issue | Detail | Handling |
|---|---|---|
| Team name variants | "Rising Pune Supergiant" vs "Supergiants" | Normalised in `data_loader.py` via `TEAM_ABBREV` map |
| 0-indexed overs | Some dataset versions use 0-indexed overs | Detected at load time, warning logged |
| Player dismissal nulls | `player_dismissed` NaN = not out | Kept as NaN; never filled with 0 |
| Super overs | Included in raw data | Excluded from all feature engineering via `is_super_over` flag |

---

<div align="center">

Built with 🏏 by [Pranav Gawale](https://github.com/PRANAVGAWALE-DS)

⭐ Star the repo if you found it useful

</div>