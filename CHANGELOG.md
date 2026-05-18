# Changelog

All notable changes to Cricket ML are documented here.
Format: `[version] — date — description`

---

## [1.3.0] — 2026-05-17 — Tier 2: GRU Score Predictor

### Added
- `src/gru_score_model.py` — PyTorch 2-layer GRU for 1st-innings score prediction.
  Ingests each over as a timestep using `pack_padded_sequence` for variable-length
  sequences. 10 input features per step (7 dynamic + 3 static context).
- `notebooks/06_gru_score_predictor.py` — full training pipeline: over sequence
  aggregation → temporal train/val split → normalisation → GRU training with early
  stopping → save as `models/gru_score_predictor.pt`.
- `POST /predict/score/gru` — new FastAPI endpoint. Accepts per-over stats
  (runs, wickets, boundaries) for 1–20 completed overs; derives cumulative features
  server-side; returns predicted score with MAE-based confidence interval.
- Streamlit Score Predictor page: two-tab layout.
  - ⚡ LightGBM tab — existing over-10 snapshot UI unchanged.
  - 🧠 GRU tab — over-by-over entry form with live cumulative table,
    prediction on demand, and a live prediction curve (GRU vs naive RR×20)
    that renders after 3+ overs.

### Model performance
| Model | MAE | R² | Note |
|---|---|---|---|
| LightGBM (static over-10) | 17.9 runs | 0.52 | Over-10 snapshot only |
| GRU (rolling sequence) | 18.3 runs | 0.33 | Any over 1–20 |

GRU enables over-by-over inference which LightGBM cannot. On 756 matches the GRU marginally underperforms due to limited sequence count — Cricsheet data (planned) expected to close the gap.

---

## [1.2.0] — 2026-05-17 — Tier 1: Rolling Player-Form Features

### Added
- `src/rolling_features.py` — leak-free rolling player-form computation.
  Per-player batting (avg, SR) and bowling (economy, bowling SR) aggregated
  to team level using a 5-match rolling window with `shift(1)` to exclude the
  current match. Cold-start fill via global dataset priors.
- `notebooks/04_rolling_features.py` — full pipeline: compute rolling form →
  save `data/processed/team_rolling_form.parquet` → build
  `data/processed/match_features_v3.parquet` (20 features vs v2's 9) →
  retrain match winner → save `models/match_winner.ubj`.
- Match winner model upgraded from v2 (9 features) to v3 (20 features).
  Rolling features dominate the top-10 importances;
  `team2_rolling_bat_avg` ranks #1.

### Model performance
| Model | Metric | Before (636 matches) | After (756 matches, v3) |
|---|---|---|---|
| Match Winner | AUC | 0.530 | **0.797** |
| Match Winner | CV AUC | — | 0.800 ± 0.045 |
| Match Winner | Accuracy | — | 0.748 |

AUC improvement: +0.267 (+50% relative). Rolling features dominate top-10 importances — `team2_rolling_bat_avg` ranks #1.

---

## [1.1.0] — 2026-05-17 — Tier 3: Serving + Interactive UX

### Added
- `api/main.py` — FastAPI app with 6 routes:
  `GET /health`, `GET /matches`,
  `POST /predict/match-winner`, `POST /predict/score`,
  `GET /predict/win-curve/{match_id}`, `POST /predict/potm`.
- `api/schemas.py` — Pydantic v2 request/response contracts for all endpoints.
- `streamlit_app.py` — 5-page Streamlit dashboard:
  🏏 Match Winner, 📊 Score Predictor, 📈 Win Probability Curve,
  🏆 POTM Predictor, 🔍 Player Stats Explorer.
- Player Stats Explorer — pure pandas aggregation from `deliveries.parquet`.
  Batting tab: career totals, dual-axis season chart (runs + SR), scoring
  breakdown donut. Bowling tab: wickets/economy per season. Leaderboard tab:
  all-time top 15 batters (≥50 balls) and bowlers (≥10 overs).
- `docker/Dockerfile.api`, `docker/Dockerfile.streamlit`,
  `docker-compose.yml` — full containerised stack.

### Fixed
- Gauge step colors: replaced 8-char hex (`#rrggbbaa`) with `rgba()` — Plotly
  rejects alpha-hex in `indicator.gauge.step.color`.
- POTM Arrow serialisation crash: `Economy` and `Strike Rate` columns are now
  uniformly `str` dtype, preventing PyArrow mixed-type error.
- `use_container_width` deprecation on `st.dataframe` calls → `width="stretch"`.
- `health["teams"]` KeyError: `HealthResponse` schema now declares all 4 fields;
  Streamlit uses `.get()` with empty-list fallbacks.
- `ReadTimeout` on first call: `REQUEST_TIMEOUT` raised to 30 s; `--reload`
  flag removed from production uvicorn invocation.

---

## [1.0.1] — 2026-05-17 — Serialisation hardening

### Fixed
- XGBoost pickle warning: `match_winner` and `potm_classifier` now saved as
  `.ubj` (XGBoost native binary JSON) via `model.save_model()`.
  `load_model()` in `src/models.py` and `_load_xgb_model()` in `api/main.py`
  resolve `.ubj` before `.pkl` with graceful fallback.
- `_post()` and `_get()` in Streamlit hardened against non-JSON HTTP 500
  responses — no longer crash with `JSONDecodeError` when the API returns
  an HTML error page.

---

## [1.0.0] — 2026-05 — Initial release

### Added
- `src/data_loader.py` — `load_matches()`, `load_deliveries()`,
  `load_both()`, `save_processed()`. Schema validation, team abbreviation,
  super-over flagging, legal delivery derivation.
- `src/features.py` — `build_match_features_v2()`, `build_score_features()`,
  `build_win_probability_features()`, `build_potm_features()`.
- `src/models.py` — `train_match_winner()`, `train_score_predictor()`,
  `train_win_probability()`, `train_potm_classifier()`, `predict_win_curve()`,
  `load_model()`.
- `notebooks/01_eda.py` — 17 Plotly charts covering score distributions,
  team win rates, toss analysis, venue effects, season trends.
- `notebooks/02_feature_engineering.py` — 6 feature validation charts.
- `notebooks/03_modeling.py` — trains all 4 base models, saves to `models/`.

### Model performance (baseline)
| Model | Metric | Value |
|---|---|---|
| Match Winner | AUC | 0.530 |
| Score Predictor | MAE | ~13 runs |
| Score Predictor | R² | 0.44 |
| Win Probability | AUC | 0.864 |
| POTM Classifier | AUC | (imbalanced — PR-AUC reported) |

Data: IPL 2008–2019, 636 matches, 150,460 ball-by-ball deliveries.