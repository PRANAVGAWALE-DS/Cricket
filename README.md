<div align="center">

# 🏏 Cricket ML — IPL Prediction Suite

**Production-grade ML pipeline for IPL analytics**  
*5 models · 7 API endpoints · Live Streamlit dashboard · Dockerised · CI/CD*

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-HuggingFace_Spaces-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/spaces/PG-AIML/Cricket)
[![GitHub Repo](https://img.shields.io/badge/GitHub-PRANAVGAWALE--DS/Cricket-181717?style=for-the-badge&logo=github)](https://github.com/PRANAVGAWALE-DS/Cricket)
[![CI](https://img.shields.io/github/actions/workflow/status/PRANAVGAWALE-DS/Cricket/ci.yml?branch=main&style=for-the-badge&label=CI)](https://github.com/PRANAVGAWALE-DS/Cricket/actions)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)](LICENSE)

<br/>

<img src="https://img.shields.io/badge/XGBoost-Match_Winner_AUC_0.797-FF6B35?style=flat-square"/>
<img src="https://img.shields.io/badge/LightGBM-Win_Probability_AUC_0.822-00B4D8?style=flat-square"/>
<img src="https://img.shields.io/badge/PyTorch_GRU-Score_Predictor-EE4C2C?style=flat-square"/>
<img src="https://img.shields.io/badge/Tests-37_passed-22C55E?style=flat-square"/>

</div>

---

## 📖 What Is This?

A complete, end-to-end ML system that turns raw IPL ball-by-ball data into live predictions. Built with production engineering practices — not a notebook project.

**What makes it different:**

- Match Winner AUC jumped **0.530 → 0.797** by engineering rolling player-form features (last-5-match batting avg, economy, bowling SR) — *without adding a single new data source*
- **GRU sequence model** replaces the static LightGBM snapshot, enabling score prediction at *any* over (1–20), not just over 10
- **7-endpoint FastAPI backend** + **5-page Streamlit dashboard** serving real-time predictions, deployed on HF Spaces
- **43 automated tests** across 7 test classes with GitHub Actions CI on every push

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Cricket ML Stack                         │
│                                                                   │
│  Raw Data          Feature Engineering        ML Models           │
│  ──────────        ───────────────────        ─────────           │
│  matches.csv  ──▶  match_features_v3   ──▶   XGBoost             │
│  deliveries   ──▶  score_features      ──▶   LightGBM  ──▶ API   │
│  .csv (756    ──▶  win_prob_features   ──▶   PyTorch              │
│   matches,    ──▶  potm_features       ──▶   GRU                  │
│  179k balls)  ──▶  rolling_form        ──▶   XGBoost              │
│                                                                   │
│  ┌─────────────────────┐  HTTP  ┌──────────────────────────┐     │
│  │  FastAPI  :8000     │◀──────▶│  Streamlit  :8501/:7860  │     │
│  │  7 endpoints        │        │  5 interactive pages      │     │
│  │  Pydantic v2        │        │  live GRU inference       │     │
│  └─────────────────────┘        └──────────────────────────┘     │
│                                                                   │
│  Docker Compose (local)  ·  HF Spaces Docker (production)        │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📊 Model Performance

| Model | Algorithm | Metric | Value | Notes |
|---|---|---|---|---|
| **Match Winner** | XGBoost | AUC | **0.797** | +0.267 from rolling features |
| **Score Predictor** | LightGBM | MAE | **18 runs** | Static over-10 snapshot |
| **GRU Score Predictor** | PyTorch GRU | MAE | **18 runs** | Any over 1–20 |
| **Win Probability** | LightGBM | AUC | **0.822** | Ball-by-ball live |
| **POTM Classifier** | XGBoost | AUC | **0.972** | PR-AUC 0.67 (imbalanced) |

### Match Winner AUC Journey

```
v1 (base, 9 features)          ████░░░░░░░░░░░░░░░░  0.530
v3 (rolling form, 20 features) ████████████████░░░░  0.797  ▲ +50% relative
```

> **Key insight:** The AUC jump came entirely from *feature engineering* — rolling per-player batting average, strike rate, economy and bowling strike rate aggregated to team level with a `shift(1)` leak guard. No new data, no architecture change, no hyperparameter search.

---

## 🗂️ Project Structure

```
Cricket/
├── 📁 api/
│   ├── main.py              FastAPI — 7 routes, lifespan startup, GRU loader
│   └── schemas.py           Pydantic v2 request/response contracts
│
├── 📁 src/
│   ├── data_loader.py       load_matches(), load_deliveries(), save_processed()
│   ├── features.py          build_*_features() — all feature engineering
│   ├── models.py            train_*() — all model training + serialisation
│   ├── rolling_features.py  Leak-free rolling player-form (shift(1) guard)
│   └── gru_score_model.py   PyTorch GRU — Dataset, training loop, save/load
│
├── 📁 notebooks/
│   ├── 01_eda.py            17 Plotly charts
│   ├── 02_feature_engineering.py
│   ├── 03_modeling.py       Base models
│   ├── 04_rolling_features.py  Rolling form → retrain match winner
│   ├── 05_fix_sklearn_warning.py
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
│   └── upload_to_hub.py     One-time HF Hub artefact upload
│
├── 📁 .github/workflows/
│   └── ci.yml               Lint → Tests → Docker build (3 jobs)
│
├── Dockerfile               HF Spaces single-container deployment
├── start.sh                 Starts FastAPI + Streamlit in same container
├── streamlit_app.py         5-page dashboard (52 KB)
├── docker-compose.yml
├── Makefile                 make train / make serve / make docker-up
├── requirements.txt
└── requirements_tier3.txt
```

---

## ⚡ Quickstart

### Prerequisites

```bash
git clone https://github.com/PRANAVGAWALE-DS/Cricket.git
cd Cricket
```

Python 3.11, Docker Desktop (optional).
Model artefacts are **not** in the repo — generate them by running the notebooks.

### Option A — Local (two terminals)

```bash
# Install dependencies
pip install -r requirements.txt
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu

# Run full training pipeline (~30 min, GRU is bottleneck)
make train

# Terminal 1 — FastAPI
set PYTHONPATH=.
uvicorn api.main:app --host 0.0.0.0 --port 8000

# Terminal 2 — Streamlit
streamlit run streamlit_app.py
```

| Service | URL |
|---|---|
| FastAPI docs | http://localhost:8000/docs |
| API health | http://localhost:8000/health |
| Dashboard | http://localhost:8501 |

### Option B — Docker Compose

```bash
docker compose up --build
# API → :8000   Dashboard → :8501
```

### Option C — Live Demo

👉 **[huggingface.co/spaces/PG-AIML/Cricket](https://huggingface.co/spaces/PG-AIML/Cricket)**

---

## 🔌 API Reference

**Base URL:** `http://localhost:8000`

<details>
<summary><code>GET /health</code> — API status + loaded models</summary>

```json
{
  "status": "ok",
  "models_loaded": { "match_winner": true, "score_predictor": true,
                     "win_probability": true, "potm_classifier": true },
  "teams": ["CSK", "DC", "KKR", "MI", "..."],
  "venues": ["Eden Gardens", "Wankhede Stadium", "..."]
}
```
</details>

<details>
<summary><code>GET /matches</code> — All match IDs available for the win-curve endpoint</summary>

```json
{
  "match_ids": [1, 2, 3, "..."],
  "count": 754
}
```
</details>

<details>
<summary><code>POST /predict/match-winner</code> — Pre-match win probability</summary>

```json
// Request
{ "team1": "MI", "team2": "CSK", "venue": "Wankhede Stadium",
  "toss_winner": "team1", "toss_decision": "bat", "season": 2019 }

// Response
{ "team1": "MI", "team2": "CSK",
  "team1_win_probability": 54.3, "team2_win_probability": 45.7 }
```
</details>

<details>
<summary><code>POST /predict/score</code> — LightGBM score predictor (over-10)</summary>

```json
// Request
{ "batting_team": "MI", "venue": "Wankhede Stadium", "season": 2019,
  "runs_10": 68, "wickets_10": 2, "boundaries_10": 9 }

// Response
{ "predicted_final_score": 174.5, "confidence_interval_low": 161.5,
  "confidence_interval_high": 187.5, "current_rr": 6.8, "projected_naive": 136.0 }
```
</details>

<details>
<summary><code>POST /predict/score/gru</code> — GRU score predictor (any over)</summary>

```json
// Request — pass completed overs in chronological order
{ "batting_team": "MI", "venue": "Wankhede Stadium", "season": 2019,
  "overs": [
    {"runs_in_over": 8, "wickets_in_over": 0, "boundaries_in_over": 2},
    {"runs_in_over": 6, "wickets_in_over": 1, "boundaries_in_over": 1}
  ]
}

// Response
{ "predicted_final_score": 168.3, "confidence_interval_low": 150.3,
  "confidence_interval_high": 186.3, "overs_seen": 2, "model": "GRU" }
```
</details>

<details>
<summary><code>GET /predict/win-curve/{match_id}</code> — Over-by-over win probability</summary>

```json
{ "match_id": 335982, "batting_team": "MI", "bowling_team": "CSK",
  "curve": [{"over": 1, "win_probability": 48.2}, {"over": 2, "win_probability": 51.7}],
  "actual_winner": "MI" }
```
</details>

<details>
<summary><code>POST /predict/potm</code> — Player of the Match predictor</summary>

```json
// Request
{ "players": [{ "player_name": "Rohit Sharma", "runs_scored": 78,
  "balls_faced": 48, "wickets_taken": 0, "runs_given": 0,
  "balls_bowled": 0, "player_won": 1 }] }

// Response
{ "predicted_potm": "Rohit Sharma",
  "players": [{"player_name": "Rohit Sharma", "potm_probability": 81.4, "rank": 1}] }
```
</details>

---

## 📱 Dashboard Pages

| Page | What it does |
|---|---|
| 🏏 **Match Winner** | Pre-match probability gauge. Uses v3 model with rolling player form. |
| 📊 **Score Predictor** | Two tabs: LightGBM (over-10) vs GRU (enter overs one-by-one, live curve) |
| 📈 **Win Probability** | Select any historical match → animated over-by-over win curve |
| 🏆 **POTM Predictor** | Enter player stats → ranked Player of the Match probabilities |
| 🔍 **Player Stats** | Career batting/bowling stats, season trends, all-time leaderboards |

---

## 🧠 Engineering Highlights

**Leak-free rolling features**
`rolling_features.py` uses `shift(1)` before the rolling window — current match stats are always excluded from the feature vector. Cold-start handled via global dataset priors.

**XGBoost native serialisation**
Models saved as `.ubj` (XGBoost's binary JSON) via `model.save_model()`. Eliminates `UserWarning` from joblib pickle version mismatches across XGBoost upgrades.

**GRU with variable-length sequences**
`pack_padded_sequence` handles innings of different lengths. Training uses a temporal split (last 20% of match IDs as validation) to simulate real-world hold-out.

**Lifespan startup (FastAPI)**
Migrated from deprecated `@app.on_event("startup")` to `@asynccontextmanager lifespan` — required for correct TestClient behaviour in pytest.

**Encoding compatibility without a stored encoder**
Models use `pd.Series.astype("category").cat.codes` for team and venue encoding. Codes are assigned in **alphabetically sorted order** of unique values seen in training. `api/main.py` reconstructs these maps at startup from the same processed parquets — inference encoding always matches training encoding without storing a separate `LabelEncoder` artefact.

**43-test CI pipeline**
7 test classes: data loader schema, feature engineering, rolling features, model serialisation, GRU, API routes, model validation gate (AUC thresholds). Smoke artefact builder generates synthetic data so CI never needs real CSVs or models.

---

## 🗺️ Roadmap

- [x] Tier 3 — FastAPI + Streamlit + Docker + Player Stats
- [x] Tier 1 — Rolling player-form features (AUC 0.530 → 0.797)
- [x] Tier 2 — GRU sequence score predictor
- [x] GitHub Actions CI (43 tests, 3 jobs)
- [x] HF Spaces Docker deployment
- [ ] Cricsheet data (IPL 2020–2024, ~5k matches) — GRU target MAE < 15
- [ ] Win Probability GRU (target AUC > 0.90)
- [ ] Fantasy points optimizer

---

## 📦 Tech Stack

| Layer | Tools |
|---|---|
| ML | XGBoost, LightGBM, PyTorch (GRU), scikit-learn |
| Data | Pandas, NumPy, PyArrow |
| API | FastAPI, Pydantic v2, Uvicorn |
| Dashboard | Streamlit, Plotly |
| DevOps | Docker, GitHub Actions, HF Spaces |
| Testing | pytest, pytest-cov, httpx (FastAPI TestClient) |

---

## 📄 Data Source

IPL ball-by-ball data: [Kaggle — nowke9/ipldata](https://www.kaggle.com/datasets/nowke9/ipldata)
Seasons: **2008–2019** · **756 matches** · **179,078 deliveries**

---

<div align="center">

Built with 🏏 by [Pranav Gawale](https://github.com/PRANAVGAWALE-DS)

⭐ Star the repo if you found it useful

</div>