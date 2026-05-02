# 🏏 IPL Cricket Analysis — Production ML Edition

A comprehensive data analysis and machine learning project on IPL cricket data (2008–2019),
upgraded from a basic EDA to a full production-grade analytics pipeline.

---

## Project Structure

```
cricket-analysis/
├── data/
│   ├── raw/                  ← put deliveries.csv + matches.csv here
│   └── processed/            ← cleaned parquet files (auto-generated)
├── notebooks/
│   ├── 01_eda.py             ← full EDA with interactive Plotly charts
│   ├── 02_feature_engineering.py
│   └── 03_modeling.py        ← 4 ML models
├── src/
│   ├── data_loader.py        ← schema-validated, path-agnostic loader
│   ├── features.py           ← all cricket feature engineering
│   └── models.py             ← model training + evaluation + serialization
├── models/                   ← saved .pkl model artifacts
└── requirements.txt
```

---

## Dataset

| File | Rows | Columns | Description |
|---|---|---|---|
| `matches.csv` | 636 | 18 | Match-level: teams, venue, toss, result |
| `deliveries.csv` | 150,460 | 21 | Ball-by-ball: batsman, bowler, runs, dismissal |

## Data
Download `matches.csv` and `deliveries.csv` from the
[IPL Dataset on Kaggle](https://www.kaggle.com/datasets/nowke9/ipldata)
and place them in `data/raw/` before running.

---

## Setup

```bash
conda create -n cricket-analysis python=3.11 -y
conda activate cricket-analysis
pip install -r requirements.txt
```

> **VS Code users:** The notebooks use `# %%` cell markers. Open any `.py` file in `notebooks/` and click **Run Cell**.

**Run order:**
```
python notebooks/01_eda.py
python notebooks/02_feature_engineering.py
python notebooks/03_modeling.py
```

---

## EDA Highlights (01_eda.py)

All charts are interactive Plotly — hover, zoom, filter.

- **Seasonal trends:** Matches per season, toss decision breakdown
- **Batting:** Career stats, strike rate vs average scatter (bubble = runs, colour = boundary %), phase-wise SR (Powerplay / Middle / Death), dot ball pressure index
- **Bowling:** Wickets, economy, bowling SR, phase-wise economy
- **Venues:** Average 1st innings score per venue, match distribution
- **Head-to-head:** Win count heatmap across all team pairs
- **Season evolution:** Has T20 batting gotten more aggressive? (run rate and six rate by season)

**Key findings:**
- Mumbai Indians lead all-time wins; Chennai Super Kings are the most consistent runners-up
- Chris Gayle has the most Player of the Match awards in IPL history
- Teams winning the toss choose to field 57% of the time — but toss-win to match-win correlation is essentially zero (see Model Limitations below)
- Death-over economy has worsened consistently from 2012 onwards — batsmen have improved faster than bowlers
- M. Chinnaswamy Stadium (Bengaluru) produces the highest average 1st innings scores; spin-friendly venues like Chepauk average ~15 runs lower

---

## ML Models

### Model 1 — Match Winner Classifier

**Task:** Predict which team wins before the match starts.

**Algorithm:** XGBoost classifier

**Features:** Rolling historical win rates per team (no leakage), toss decision, venue, team encodings, season

| Metric | Value |
|---|---|
| CV AUC (5-fold) | 0.530 |
| Test AUC | ~0.50–0.53 |
| Accuracy | ~52% |

**Key finding — and why it is correct:**
The strongest feature is `win_rate_diff` (correlation 0.080) — the difference in each team's historical win rate going into the match. Toss outcome and venue carry near-zero signal individually (correlation < 0.015). This is not a model failure; it reflects a genuine property of T20 cricket. With only 636 total matches and 13 teams across 12 seasons, pre-match prediction is an inherently low-signal problem. The realistic ceiling for this feature set is **AUC 0.55–0.58**; going beyond that would require player availability, pitch reports, and head-to-head records at the specific venue — data not present in this dataset.

> This is a real finding worth reporting: toss does not significantly predict IPL match outcomes.

---

### Model 2 — First Innings Score Predictor

**Task:** Predict the final 1st innings score at the halfway point (end of over 10).

**Algorithm:** LightGBM regressor

**Prediction point:** Over 10 — this is the "drinks break" calculation teams actually make in the middle.

**Features:** Runs at over 10, wickets fallen, current run rate, projected score (linear), scoring pressure vs venue historical average, boundaries hit, batting team, venue, season

| Metric | Value |
|---|---|
| MAE | ~13–15 runs |
| R² | ~0.50–0.65 |

**Why over-10 instead of over-6:**
The powerplay (overs 1–6) snapshot only captures 30% of the innings. At over 10, the model has seen half the innings, including how the middle overs opened. The key new feature, `scoring_pressure`, measures how far above or below the venue's historical run rate the batting team is scoring — this captures match context that raw runs cannot.

---

### Model 3 — Live Win Probability ✅ Best Model

**Task:** Predict the batting team's probability of winning at each over in the 2nd innings.

**Algorithm:** LightGBM classifier

**Training:** Temporal split — last 20% of match IDs held out as test set (prevents leakage from future matches).

| Metric | Value |
|---|---|
| AUC | **0.864** |
| Accuracy | 77.2% |
| Log Loss | 0.454 |

**Feature importance order:** `runs_required` > `balls_remaining` > `required_rr` > `wickets_fallen` > `current_rr`

**Example inference:**
> *Over 15, batting team has scored 110/3, needs 60 more off 30 balls (required RR = 12.0)*
> → **Predicted win probability: 31.5%** ✓ (intuitive — high asking rate with 3 wickets down)

This is the most portfolio-ready model. The win probability curve across 20 overs is visually compelling and directly interpretable.

---

### Model 4 — Player of the Match Classifier

**Task:** Predict which player will win the POTM award given their match performance.

**Algorithm:** XGBoost with `scale_pos_weight` for class imbalance (4.75% positive rate).

| Metric | Value | Note |
|---|---|---|
| ROC-AUC | 0.967 | Inflated by easy negatives |
| **PR-AUC** | **0.615** | Correct metric for imbalanced classification |
| Precision | 35.7% | 1 in 3 predicted POTMs are correct |
| Recall | 91.3% | Catches 9 out of 10 actual POTMs |
| F1 | 0.513 | |

**Why accuracy (91.75%) is misleading here:**
A model predicting "not POTM" for every player would achieve 95.25% accuracy. PR-AUC of 0.615 vs a random baseline of 0.0475 shows the model is genuinely learning signal. The high recall (0.913) means almost every actual POTM is in the model's top predictions — the issue is specificity, not sensitivity.

**Feature importance order:** `runs_scored` > `player_won` > `wickets_taken` > `strike_rate` > `economy`

> **Finding:** A POTM award is 4× more likely when the player's team wins. Run scoring is the dominant individual signal, outweighing bowling performance even for specialist bowlers.

---

## Design Decisions

| Decision | Rationale |
|---|---|
| No hardcoded paths | `Path(__file__).resolve().parents[N]` — runs on any machine |
| `fillna(0)` replaced | `player_dismissed` NaN means "not out"; only numeric extras get zero-filled |
| `.astype(bool)` before `~` | Prevents numpy memory errors after parquet dtype drift; also fixes `~a & b` precedence bug |
| Temporal train/test split | Win probability uses last 20% of match IDs, not random split — prevents future leakage |
| `scale_pos_weight` for POTM | Correct handling of 20:1 class imbalance; naive accuracy is not reported as primary metric |
| Rolling features computed row-by-row | `win_rate_diff`, `team_venue_avg_score`, `venue_avg_rr` all use strictly prior-match history |

---

## Technologies

| Layer | Stack |
|---|---|
| Data | pandas, numpy |
| Visualisation | Plotly (interactive) |
| ML | XGBoost, LightGBM, scikit-learn |
| Serialisation | joblib, parquet |
| Environment | Conda, Python 3.11 |