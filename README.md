# 🏏 IPL Cricket Analysis
### Production-Grade Machine Learning on 12 Seasons of Ball-by-Ball Data

![XGBoost](https://img.shields.io/badge/XGBoost-Classifier-FF6600?style=flat)
![LightGBM](https://img.shields.io/badge/LightGBM-Regressor%20%7C%20Classifier-02569B?style=flat)
![Plotly](https://img.shields.io/badge/Plotly-Interactive%20Charts-3F4F75?style=flat)
![pandas](https://img.shields.io/badge/pandas-Data%20Processing-150458?style=flat)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

*From raw CSVs to four production ML models — zero notebooks, zero shortcuts.*

---

## What This Project Does

This project turns 150,460 ball-by-ball IPL deliveries (2008–2019) into four production-grade ML models, each solving a distinct cricket analytics problem:

| # | Model | Task | Algorithm | Best Metric |
|---|-------|------|-----------|-------------|
| 1 | **Match Winner** | Pre-match outcome prediction | XGBoost | AUC 0.501 |
| 2 | **Score Predictor** | Final 1st innings total at over 10 | LightGBM | MAE 16 runs |
| 3 | **Live Win Probability** ⭐ | Ball-by-ball chase probability | LightGBM | **AUC 0.864** |
| 4 | **POTM Classifier** | Player of the Match prediction | XGBoost | PR-AUC 0.641 |

> **⭐ Headline model:** The live win probability model achieves AUC 0.864 with a temporal train/test split — predicting match outcomes over-by-over with no data leakage from future matches.

---

## Project Structure

```
cricket-analysis/
│
├── data/
│   ├── raw/                        ← matches.csv + deliveries.csv (from Kaggle)
│   └── processed/                  ← auto-generated parquet files
│
├── notebooks/
│   ├── 01_eda.py                   ← 18 interactive Plotly charts
│   ├── 02_feature_engineering.py   ← feature builds + 6 validation charts
│   ├── 03_modeling.py              ← 4 models + 6 evaluation charts
│   └── .charts/                    ← auto-generated HTML chart files
│       ├── eda_01.html … eda_17.html
│       ├── features_01.html … features_06.html
│       └── modeling_01.html … modeling_06.html
│
├── src/
│   ├── data_loader.py              ← schema-validated, path-agnostic CSV loader
│   ├── features.py                 ← all feature engineering (batting, bowling,
│   │                                  venue, partnerships, ML feature builders)
│   └── models.py                   ← training, evaluation, serialization
│
├── models/                         ← serialized .pkl artifacts (joblib)
│   ├── match_winner.pkl
│   ├── score_predictor.pkl
│   ├── win_probability.pkl
│   └── potm_classifier.pkl
│
└── requirements.txt
```

---

## Dataset

Download from [Kaggle — IPL Dataset (nowke9)](https://www.kaggle.com/datasets/nowke9/ipldata) and place both files in `data/raw/`.

| File | Rows | Columns | Granularity |
|------|------|---------|-------------|
| `matches.csv` | 636 | 18 | One row per match |
| `deliveries.csv` | 150,460 | 23 | One row per ball |

---

## Setup

```bash
conda create -n cricket-analysis python=3.11 -y
conda activate cricket-analysis
pip install -r requirements.txt
```

> **VS Code:** Files use `# %%` cell markers — open any notebook and click **Run Cell** to run interactively. Charts open as browser tabs automatically.

**Run in order:**

```bash
python notebooks/01_eda.py                 # EDA → 18 charts in .charts/
python notebooks/02_feature_engineering.py # Features → 6 charts + 4 parquet files
python notebooks/03_modeling.py            # Models → 6 charts + 4 .pkl files
```

---

## EDA — 18 Interactive Charts (`01_eda.py`)

All charts are written to `notebooks/.charts/eda_*.html` and opened in the browser via `file://` (no local server dependency). Each chart is interactive — hover, zoom, filter.

### Match-Level Trends
| Chart | What it shows |
|-------|---------------|
| Matches per season | IPL grew from 58 matches in 2008 to a peak in 2013; rain-affected seasons visible |
| Toss decision by season | Field-first preference climbed from ~50% in 2008 to 70%+ by 2015 |
| Toss winner = match winner? | 50.4% — effectively a coin flip; toss has zero predictive power |
| Top 20 venues | Wankhede and Eden Gardens host the most matches |
| Top 15 POTM | Chris Gayle leads with the most Player of the Match awards in IPL history |
| Team win counts | Mumbai Indians lead all-time; CSK most consistent despite two-year ban |

### Batting Analytics
| Chart | What it shows |
|-------|---------------|
| Strike rate vs average scatter | Bubble = career runs, colour = boundary %; separates impact players from volume scorers |
| Top 15 run scorers | Bar chart coloured by strike rate |
| Phase-wise strike rate | Powerplay / Middle / Death breakdown for top 20 batsmen |
| Dot ball pressure index | Which batsmen are most susceptible to pressure bowling |

### Bowling Analytics
| Chart | What it shows |
|-------|---------------|
| Economy vs strike rate scatter | Bubble = wickets, colour = dot ball %; identifies true match-winners |
| Top 15 wicket takers | Bar chart coloured by economy |
| Phase-wise economy | Death-over economy has worsened every season since 2012 |

### Venue & Context
| Chart | What it shows |
|-------|---------------|
| Avg 1st innings by venue | Chinnaswamy (+15 above average); Chepauk (-15 below) |
| Head-to-head win matrix | Full heatmap of all team vs team win counts |
| Super over results | Runs vs wickets in all 7 super overs |
| Run rate + six rate trend | T20 batting has measurably evolved — both metrics trend upward |

---

## ML Models

### Model 1 — Match Winner Classifier

**Question answered:** *Who wins before the first ball is bowled?*

**Feature set** (all computed from prior matches only — no leakage):

```
win_rate_diff      ← strongest signal (corr = 0.080)
win_rate_team1     ← rolling historical win rate, match N uses only matches 0..N-1
win_rate_team2
toss_winner_is_team1
bat_first
venue_enc
team1_enc / team2_enc
season
```

| Metric | Value |
|--------|-------|
| CV AUC (5-fold) | 0.530 |
| Test AUC | 0.501 |
| Accuracy | 52.8% |

**The honest finding:** This is not a model failure. T20 cricket is genuinely unpredictable before the match starts. The realistic AUC ceiling for this feature set is **0.55–0.58** — reaching it would require player availability, pitch reports, and weather data that don't exist in this dataset. Reporting AUC 0.501 is the correct thing to do.

---

### Model 2 — First Innings Score Predictor

**Question answered:** *At the drinks break (over 10), what will the final total be?*

**Why over 10 instead of over 6:** The powerplay snapshot captures only 30% of the innings. At over 10, the model has seen half the match including how the middle overs opened — the most tactically variable phase in T20.

**Key engineered feature — `scoring_pressure`:**
```
scoring_pressure = current_rr − venue_historical_avg_rr
```
Captures whether the batting team is ahead of or behind the venue's historical pace. The same score of 80/2 at over 10 means very different things at Chinnaswamy vs Chepauk.

| Metric | Value |
|--------|-------|
| MAE | **16.0 runs** |
| R² | 0.44 |

---

### Model 3 — Live Win Probability ⭐

**Question answered:** *Right now, over by over in the second innings, what are the chasing team's odds?*

**Training discipline:** Match IDs sorted chronologically before the 80/20 split. `pd.unique()` returns by first-appearance order — not chronological — so the naive version was not a genuine future hold-out. Fixed with `np.sort()` before slicing.

**Feature importance:**
```
1. runs_required     ← how many still needed
2. balls_remaining   ← runway left
3. required_rr       ← asking rate
4. wickets_fallen    ← resources consumed
5. current_rr        ← momentum
```

| Metric | Value |
|--------|-------|
| AUC | **0.864** |
| Accuracy | 77.2% |
| Log Loss | 0.454 |

**Live inference example:**
```
Over 15 | 110/3 | Need 60 off 30 balls | Required RR = 12.0
→ Win probability: 25.2%  ✓ (correctly pessimistic — high ask, 3 wickets down)
```

---

### Model 4 — Player of the Match Classifier

**Question answered:** *Given a player's match performance, how likely are they to win POTM?*

**Class imbalance:** 633 POTM awards across 13,327 player-match records = **4.75% positive rate**. Handled with `scale_pos_weight`; accuracy (92.4%) is meaningless here — a "predict never" baseline scores 95.25%.

**Feature importance:**
```
1. runs_scored    ← batting dominates POTM selection
2. player_won     ← 4× more likely if on the winning team
3. wickets_taken  ← bowling impact, but secondary to runs
4. strike_rate
5. economy
```

| Metric | Value | Note |
|--------|-------|------|
| ROC-AUC | 0.970 | Inflated by easy negatives — not the primary metric |
| **PR-AUC** | **0.641** | Correct metric; random baseline = 0.0475 |
| Precision | 37.9% | Better than 1 in 3 predictions correct |
| Recall | 92.9% | Catches 9 out of 10 actual POTM winners |
| F1 | 0.539 | |

---

## Engineering Notes

### Data Integrity Fixes Applied

| Bug | Impact | Fix |
|-----|--------|-----|
| Dot balls counted on wide deliveries | `dot_ball_pct` overstated for every batsman | Filter to `is_legal_delivery` before aggregating |
| Temporal split used unsorted `unique()` | Test set not genuinely future matches | `np.sort()` before slicing last 20% |
| Pure bowlers got `player_won = 0` | POTM model penalised winning bowlers who never batted | Unified team lookup from both batting and bowling deliveries |
| City fill extracted stadium name | `str[0]` → "Wankhede Stadium" not "Mumbai" | `str[-1]` takes last comma token |
| `use_label_encoder=False` in XGBoost | `TypeError` on XGBoost ≥ 1.6 | Parameter removed |
| Bowling match count from raw deliveries | Super-over appearances inflated match counts | Same filtered frame used for all bowling aggregates |

### Architecture Decisions

| Decision | Rationale |
|----------|-----------|
| **All feature builders in `features.py`** | Pure data transforms belong in one layer; `models.py` handles only training logic |
| **No hardcoded paths** | `Path(__file__).resolve().parents[N]` — reproducible on any machine |
| **Rolling features row-by-row** | `win_rate_diff` and `venue_avg_rr` use strictly prior-match history — no leakage |
| **`file://` chart rendering** | Plotly's local HTTP server caused `ERR_CONNECTION_REFUSED` on rapid sequential calls; `write_html` + `webbrowser.open` eliminates the server entirely |
| **Per-notebook chart prefixes** | `eda_`, `features_`, `modeling_` prefixes prevent overwrites when all three scripts run in sequence |
| **Over-indexing validated at load time** | `load_deliveries` warns immediately if `over.min() == 0`; all downstream calculations assume 1-indexed overs |

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| **Data processing** | pandas, NumPy |
| **Machine learning** | XGBoost, LightGBM, scikit-learn |
| **Visualisation** | Plotly (30 interactive charts across 3 notebooks) |
| **Serialisation** | joblib (`.pkl`), Apache Parquet |
| **Environment** | Conda, Python 3.11 |

---

## Key Findings

- **Toss is noise.** Winning the toss converts to winning the match at almost exactly 50.4%. It is the most-discussed pre-match variable in cricket and one of the least predictive.
- **Death overs favour batsmen, increasingly.** Economy in overs 16–20 has risen every season since 2012 — batting skill has outpaced bowling adaptation.
- **Chinnaswamy inflates scores by ~15 runs.** Controlling for venue is the single biggest environmental adjustment in any IPL prediction model.
- **Runs win POTM, not wickets.** Even accounting for bowling performances, run scoring is the dominant POTM predictor — specialist bowlers are systematically undervalued by the award.
- **Win probability is non-linear.** The model's curve drops sharply once required RR crosses 12 — consistent with how professional analysts assess chase difficulty in the field.