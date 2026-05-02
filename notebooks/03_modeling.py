# %% [markdown]
# # 03 · Machine Learning Models
# **IPL Cricket · Four production-grade ML models**
#
# Models trained here:
# 1. **Match Winner Classifier** — XGBoost pre-match prediction
# 2. **First Innings Score Regressor** — LightGBM after powerplay
# 3. **Live Win Probability** — LightGBM ball-by-ball
# 4. **Player of the Match Classifier** — XGBoost
#
# All models are saved to `models/` via joblib.

# %% [markdown]
# ## Setup

# %%
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.models import (
    train_match_winner,
    train_score_predictor,
    train_win_probability,
    train_potm_classifier,
    predict_win_curve,
)

TEMPLATE = "plotly_dark"
PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"

# %% [markdown]
# ## 1. Match Winner Classifier

# %%
match_feats = pd.read_parquet(PROCESSED_DIR / "match_features.parquet")
print(f"Training on {len(match_feats)} matches")

model_winner, metrics_winner = train_match_winner(match_feats)
print("\n=== Match Winner Model Metrics ===")
for k, v in metrics_winner.items():
    if k != "feature_importances":
        print(f"  {k}: {v}")

# %%
# Feature importance
fi = pd.DataFrame({
    "feature": list(metrics_winner["feature_importances"].keys()),
    "importance": list(metrics_winner["feature_importances"].values()),
}).sort_values("importance", ascending=True)

fig = px.bar(
    fi, x="importance", y="feature",
    orientation="h",
    title="Match Winner Model — Feature Importances",
    color="importance", color_continuous_scale="Viridis",
    template=TEMPLATE,
)
fig.update_layout(coloraxis_showscale=False)
fig.show()

# %% [markdown]
# ## 2. First Innings Score Regressor

# %%
score_feats = pd.read_parquet(PROCESSED_DIR / "score_features.parquet")
print(f"Training on {len(score_feats)} innings")

model_score, metrics_score = train_score_predictor(score_feats)
print("\n=== Score Predictor Metrics ===")
for k, v in metrics_score.items():
    if k != "feature_importances":
        print(f"  {k}: {v}")

# %%
# Predicted vs actual
from sklearn.model_selection import train_test_split
X_s = score_feats.drop(columns=["final_score"])
y_s = score_feats["final_score"]
_, X_test_s, _, y_test_s = train_test_split(X_s, y_s, test_size=0.2, random_state=42)
preds_s = model_score.predict(X_test_s)

fig = px.scatter(
    x=y_test_s, y=preds_s,
    labels={"x": "Actual Score", "y": "Predicted Score"},
    title=f"Score Predictor: Predicted vs Actual (MAE={metrics_score['mae']}, R²={metrics_score['r2']})",
    template=TEMPLATE,
    opacity=0.6,
)
fig.add_shape(
    type="line",
    x0=y_test_s.min(), y0=y_test_s.min(),
    x1=y_test_s.max(), y1=y_test_s.max(),
    line=dict(color="orange", dash="dash"),
)
fig.show()

# %% [markdown]
# ## 3. Live Win Probability Model

# %%
win_prob_feats = pd.read_parquet(PROCESSED_DIR / "win_prob_features.parquet")
print(f"Training on {len(win_prob_feats)} over-snapshots from {win_prob_feats['match_id'].nunique()} matches")

model_wp, metrics_wp = train_win_probability(win_prob_feats)
print("\n=== Win Probability Model Metrics ===")
for k, v in metrics_wp.items():
    if k != "feature_importances":
        print(f"  {k}: {v}")

# %%
# Feature importances
fi_wp = pd.DataFrame({
    "feature": list(metrics_wp["feature_importances"].keys()),
    "importance": list(metrics_wp["feature_importances"].values()),
}).sort_values("importance", ascending=True)

fig = px.bar(
    fi_wp, x="importance", y="feature",
    orientation="h",
    title="Win Probability Model — Feature Importances",
    color="importance", color_continuous_scale="Plasma",
    template=TEMPLATE,
)
fig.update_layout(coloraxis_showscale=False)
fig.show()

# %% [markdown]
# ### 3a. Win Probability Curve — Single Match Demo

# %%
# Pick a match from the test set to visualise the win probability curve
sample_match_ids = win_prob_feats["match_id"].unique()
demo_match_id = sample_match_ids[-5]  # pick one from the test set

try:
    curve = predict_win_curve(model_wp, demo_match_id, win_prob_feats)
    match_info = win_prob_feats[win_prob_feats["match_id"] == demo_match_id].iloc[0]
    batting_won = bool(match_info["batting_team_won"])

    fig = go.Figure()

    # Win probability curve
    fig.add_trace(go.Scatter(
        x=curve["over"], y=curve["win_probability"],
        mode="lines+markers",
        name="Batting Team Win %",
        line=dict(color="#00CC96", width=3),
        marker=dict(size=8),
        fill="tozeroy",
        fillcolor="rgba(0, 204, 150, 0.15)",
    ))

    # 50% line
    fig.add_hline(y=50, line_dash="dash", line_color="grey", opacity=0.6)

    # Outcome annotation
    outcome_text = "Batting Team WON ✓" if batting_won else "Batting Team LOST ✗"
    outcome_color = "#00CC96" if batting_won else "#EF553B"
    fig.add_annotation(
        x=curve["over"].max(), y=curve["win_probability"].iloc[-1],
        text=outcome_text,
        font=dict(color=outcome_color, size=14),
        showarrow=True, arrowcolor=outcome_color,
    )

    fig.update_layout(
        title=f"Live Win Probability Curve — Match ID {demo_match_id}",
        xaxis_title="Over",
        yaxis_title="Batting Team Win Probability (%)",
        yaxis=dict(range=[0, 100]),
        template=TEMPLATE,
    )
    fig.show()

except Exception as e:
    print(f"Could not generate curve: {e}")

# %% [markdown]
# ### 3b. Average win probability curve by outcome

# %%
drop_cols = ["batting_team_won", "match_id"]
X_wp = win_prob_feats.drop(columns=[c for c in drop_cols if c in win_prob_feats.columns])

win_prob_feats = win_prob_feats.copy()
win_prob_feats["predicted_prob"] = model_wp.predict_proba(X_wp)[:, 1] * 100

avg_by_over = (
    win_prob_feats.groupby(["over", "batting_team_won"])["predicted_prob"]
    .mean()
    .reset_index()
)
avg_by_over["outcome"] = avg_by_over["batting_team_won"].map({1: "Won", 0: "Lost"})

fig = px.line(
    avg_by_over, x="over", y="predicted_prob", color="outcome",
    title="Average Predicted Win Probability by Over — Won vs Lost Matches",
    labels={"predicted_prob": "Model Win Probability (%)", "over": "Over"},
    color_discrete_map={"Won": "#00CC96", "Lost": "#EF553B"},
    template=TEMPLATE,
)
fig.add_hline(y=50, line_dash="dash", line_color="grey", opacity=0.5)
fig.update_layout(yaxis=dict(range=[0, 100]))
fig.show()

# %% [markdown]
# ## 4. Player of the Match Classifier

# %%
potm_feats = pd.read_parquet(PROCESSED_DIR / "potm_features.parquet")
print(f"Training on {len(potm_feats)} player-match records")
print(f"POTM instances: {potm_feats['is_potm'].sum()}")

model_potm, metrics_potm = train_potm_classifier(potm_feats)
print("\n=== POTM Classifier Metrics ===")
for k, v in metrics_potm.items():
    if k != "feature_importances":
        print(f"  {k}: {v}")

# %%
fi_potm = pd.DataFrame({
    "feature": list(metrics_potm["feature_importances"].keys()),
    "importance": list(metrics_potm["feature_importances"].values()),
}).sort_values("importance", ascending=True)

fig = px.bar(
    fi_potm, x="importance", y="feature",
    orientation="h",
    title="POTM Classifier — Feature Importances",
    color="importance", color_continuous_scale="Cividis",
    template=TEMPLATE,
)
fig.update_layout(coloraxis_showscale=False)
fig.show()

# %% [markdown]
# ## 5. Model Summary Dashboard

# %%
summary = pd.DataFrame([
    {"Model": "Match Winner", "Type": "Classifier", "Algorithm": "XGBoost",
     "Key Metric": f"AUC = {metrics_winner['roc_auc']}", "Accuracy": metrics_winner['accuracy']},
    {"Model": "Score Predictor", "Type": "Regressor", "Algorithm": "LightGBM",
     "Key Metric": f"MAE = {metrics_score['mae']} runs", "Accuracy": metrics_score['r2']},
    {"Model": "Win Probability", "Type": "Classifier", "Algorithm": "LightGBM",
     "Key Metric": f"AUC = {metrics_wp['roc_auc']}", "Accuracy": metrics_wp['accuracy']},
    {"Model": "POTM Predictor", "Type": "Classifier", "Algorithm": "XGBoost",
     "Key Metric": f"AUC = {metrics_potm['roc_auc']}", "Accuracy": metrics_potm['accuracy']},
])

print("=" * 65)
print("MODEL SUMMARY")
print("=" * 65)
print(summary.to_string(index=False))
print("\nAll models saved to models/ directory.")

# %% [markdown]
# ## 6. Inference Example — Single Match Prediction

# %%
# Example: predict win probability given a match state
import joblib

model_loaded = joblib.load(MODELS_DIR / "win_probability.pkl")

# Manually constructed match state (over 15, batting team has 110/3, need 60 off 30 balls)
example_state = pd.DataFrame([{
    "over": 15,
    "runs_scored": 110,
    "wickets_fallen": 3,
    "current_rr": 110 / (15 / 6) if 15 > 0 else 0,  # approx
    "required_rr": 60 / (30 / 6),
    "balls_remaining": 30,
    "runs_required": 60,
    "venue_enc": 5,          # arbitrary encoded venue
    "batting_team_enc": 2,   # arbitrary encoded team
}])

prob = model_loaded.predict_proba(example_state)[0][1]
print(f"\nExample state: Over 15, 110/3, need 60 off 30 balls")
print(f"→ Predicted batting team win probability: {prob*100:.1f}%")
