# %% [markdown]
# # 02 · Feature Engineering
# **IPL Cricket · Building ML-ready features from ball-by-ball data**
#
# This notebook:
# - Loads cleaned parquet files from data/processed/
# - Builds match-level features for win prediction
# - Builds ball-by-ball features for live win probability
# - Builds player-level features for POTM prediction
# - Builds powerplay snapshot for score regression
# - Validates and exports all feature sets to data/processed/

# %% [markdown]
# ## Setup

# %%
import sys
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

# Write each chart to a self-contained HTML file and open via file://
# Avoids Plotly's temporary local HTTP server (ERR_CONNECTION_REFUSED).
pio.renderers.default = "browser"

_CHART_DIR = Path(__file__).resolve().parent / ".charts"
_CHART_DIR.mkdir(exist_ok=True)
_chart_counter = 0


def show(fig: go.Figure) -> None:
    """Write figure to HTML and open via file:// — no local server required."""
    global _chart_counter
    _chart_counter += 1
    out = _CHART_DIR / f"features_chart_{_chart_counter:02d}.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    webbrowser.open(out.as_uri())


from src.features import (
    build_match_features_v2,
    build_win_probability_features,
    build_score_features,
    build_potm_features,
)
from src.data_loader import save_processed

TEMPLATE = "plotly_dark"

# %% [markdown]
# ## 1. Load Processed Data

# %%
PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
matches = pd.read_parquet(PROCESSED_DIR / "matches.parquet")
deliveries = pd.read_parquet(PROCESSED_DIR / "deliveries.parquet")
print(f"matches   : {matches.shape}")
print(f"deliveries: {deliveries.shape}")

# %% [markdown]
# ## 2. Match-Level Features (for win prediction)

# %%
match_feats = build_match_features_v2(matches)
print(f"Match features shape: {match_feats.shape}")
print(
    f"Target distribution:\n{match_feats['team1_won'].value_counts(normalize=True).round(3)}"
)
match_feats.head()

# %%
# Correlation with target
corr = (
    match_feats.corr()["team1_won"]
    .drop("team1_won")
    .sort_values(key=abs, ascending=False)
)
print("Feature correlations with team1_won:")
print(corr.to_string())

# %%
save_processed(match_feats, "match_features")
print("Saved match_features.parquet")

# %% [markdown]
# ## 3. Live Win Probability Features (ball-by-ball)

# %%
win_prob_feats = build_win_probability_features(deliveries, matches)
print(f"Win probability features shape: {win_prob_feats.shape}")
print(
    f"Target balance:\n{win_prob_feats['batting_team_won'].value_counts(normalize=True).round(3)}"
)
win_prob_feats.head()

# %%
# Quick sanity: what is the win probability at over=1 vs over=19?
win_by_over = win_prob_feats.groupby("over")["batting_team_won"].mean().reset_index()
win_by_over.columns = ["over", "raw_win_rate"]

fig = px.line(
    win_by_over,
    x="over",
    y="raw_win_rate",
    title="Raw Win Rate by Over (2nd innings) — validates feature quality",
    labels={"raw_win_rate": "Win Rate (batting team)", "over": "Over"},
    template=TEMPLATE,
)
fig.add_hline(y=0.5, line_dash="dash", line_color="grey", opacity=0.5)
show(fig)

# %%
save_processed(win_prob_feats, "win_prob_features")
print("Saved win_prob_features.parquet")

# %% [markdown]
# ## 4. Score Prediction Features

# %%
score_feats = build_score_features(deliveries, matches)
print(f"Score features shape: {score_feats.shape}")
print("\nFirst innings score distribution:")
print(score_feats["final_score"].describe().round(2))

# %%
fig = px.histogram(
    score_feats,
    x="final_score",
    nbins=40,
    title="Distribution of 1st Innings Scores",
    labels={"final_score": "Score", "count": "Frequency"},
    color_discrete_sequence=["#636EFA"],
    template=TEMPLATE,
)
fig.add_vline(
    x=score_feats["final_score"].median(),
    line_dash="dash",
    line_color="orange",
    annotation_text=f"Median: {score_feats['final_score'].median():.0f}",
)
show(fig)

# %%
# Over-10 runs vs final score — colored by wickets at halfway
fig = px.scatter(
    score_feats,
    x="runs_10",
    y="final_score",
    color="wickets_10",
    title="Runs at Over 10 vs Final Score (color = wickets fallen)",
    labels={"runs_10": "Runs at Over 10", "final_score": "Final Score"},
    color_continuous_scale="RdYlGn_r",
    template=TEMPLATE,
)
slope, intercept = np.polyfit(score_feats["runs_10"], score_feats["final_score"], 1)
x_line = np.array([score_feats["runs_10"].min(), score_feats["runs_10"].max()])
fig.add_trace(
    go.Scatter(
        x=x_line,
        y=slope * x_line + intercept,
        mode="lines",
        name="Trendline",
        line=dict(color="orange", dash="dash"),
    )
)
show(fig)

# %%
# Scoring pressure vs final score — does batting above venue par translate to big totals?
fig = px.scatter(
    score_feats,
    x="scoring_pressure",
    y="final_score",
    color="wickets_10",
    title="Scoring Pressure at Over 10 vs Final Score",
    labels={
        "scoring_pressure": "RR above venue avg (pressure)",
        "final_score": "Final Score",
    },
    color_continuous_scale="RdYlGn_r",
    template=TEMPLATE,
)
fig.add_vline(
    x=0,
    line_dash="dash",
    line_color="grey",
    opacity=0.5,
    annotation_text="Venue avg RR",
)
show(fig)

# %%
save_processed(score_feats, "score_features")
print("Saved score_features.parquet")

# %% [markdown]
# ## 5. POTM Features

# %%
potm_feats = build_potm_features(deliveries, matches)
print(f"POTM features shape: {potm_feats.shape}")
print("\nClass balance (1 = Player of the Match):")
print(potm_feats["is_potm"].value_counts(normalize=True).round(4))

# %%
# Runs scored distribution: POTM vs non-POTM
fig = px.box(
    potm_feats,
    x="is_potm",
    y="runs_scored",
    title="Runs Scored: POTM (1) vs Others (0)",
    labels={"is_potm": "Is POTM", "runs_scored": "Runs Scored"},
    color="is_potm",
    template=TEMPLATE,
)
show(fig)

# %%
fig = px.box(
    potm_feats,
    x="is_potm",
    y="wickets_taken",
    title="Wickets Taken: POTM (1) vs Others (0)",
    labels={"is_potm": "Is POTM", "wickets_taken": "Wickets"},
    color="is_potm",
    template=TEMPLATE,
)
show(fig)

# %%
save_processed(potm_feats, "potm_features")
print("Saved potm_features.parquet")

# %% [markdown]
# ## Summary
# | Feature Set | Shape | Saved as |
# |---|---|---|
# | Match features | `match_feats.shape` | match_features.parquet |
# | Win prob features | `win_prob_feats.shape` | win_prob_features.parquet |
# | Score features | `score_feats.shape` | score_features.parquet |
# | POTM features | `potm_feats.shape` | potm_features.parquet |
# All feature sets are ready for model training in notebook 03.
