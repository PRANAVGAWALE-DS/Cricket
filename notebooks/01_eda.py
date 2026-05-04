# %% [markdown]
# # 01 · Exploratory Data Analysis
# **IPL Cricket · Full EDA using matches + deliveries**
#
# This notebook covers:
# - Data loading and schema validation
# - Match-level trends (seasons, venues, toss)
# - Deliveries-based batting and bowling analytics
# - Phase-wise analysis (Powerplay / Middle / Death overs)
# - Head-to-head team matrix
# - Venue par scores
# All charts use Plotly for interactivity.

# %% [markdown]
# ## Setup

# %%
import sys
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1])
)  # allow `from src import ...`

import webbrowser
from pathlib import Path as _Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

# Write each chart to a self-contained HTML file and open via file://
# This avoids Plotly's temporary local HTTP server entirely, which was
# causing ERR_CONNECTION_REFUSED on ~4 of 18 charts when servers raced.
pio.renderers.default = "browser"

_CHART_DIR = _Path(__file__).resolve().parent / ".charts"
_CHART_DIR.mkdir(exist_ok=True)
_chart_counter = 0


def show(fig: go.Figure) -> None:
    """
    Write the figure to a self-contained HTML file and open it via file://.

    fig.write_html() embeds Plotly.js from CDN and requires no local server,
    so all 18 charts open immediately with no ERR_CONNECTION_REFUSED errors.
    Files are written to notebooks/.charts/chart_01.html ... chart_18.html
    and overwritten on each run.
    """
    global _chart_counter
    _chart_counter += 1
    out = _CHART_DIR / f"eda_chart_{_chart_counter:02d}.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    webbrowser.open(out.as_uri())


from src.data_loader import load_both, save_processed
from src.features import (
    batting_career_stats,
    batting_phase_stats,
    bowling_career_stats,
    bowling_phase_economy,
    venue_par_scores,
    head_to_head_matrix,
)

pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", "{:.2f}".format)

TEMPLATE = "plotly_dark"  # swap to "plotly_white" if you prefer light mode

# %% [markdown]
# ## 1. Load Data

# %%
matches, deliveries = load_both()
print(f"matches   : {matches.shape}")
print(f"deliveries: {deliveries.shape}")
matches.head(3)

# %%
deliveries.head(3)

# %%
# Save cleaned versions for downstream notebooks
save_processed(matches, "matches")
save_processed(deliveries, "deliveries")

# %% [markdown]
# ## 2. Match-Level Trends

# %% [markdown]
# ### 2a. Matches per season

# %%
season_counts = matches.groupby("season").size().reset_index(name="matches")

fig = px.bar(
    season_counts,
    x="season",
    y="matches",
    title="IPL Matches Per Season",
    labels={"season": "Season", "matches": "Number of Matches"},
    color="matches",
    color_continuous_scale="Viridis",
    template=TEMPLATE,
)
fig.update_layout(coloraxis_showscale=False)
show(fig)

# %% [markdown]
# ### 2b. Toss decision by season

# %%
toss_season = (
    matches.groupby(["season", "toss_decision"]).size().reset_index(name="count")
)

fig = px.bar(
    toss_season,
    x="season",
    y="count",
    color="toss_decision",
    barmode="group",
    title="Toss Decision (Bat / Field) by Season",
    labels={"count": "Matches", "toss_decision": "Decision"},
    color_discrete_map={"bat": "#00CC96", "field": "#EF553B"},
    template=TEMPLATE,
)
show(fig)

# %% [markdown]
# ### 2c. Does winning the toss help?

# %%
total = len(matches.dropna(subset=["winner"]))
toss_won = (matches["toss_winner"] == matches["winner"]).sum()

fig = go.Figure(
    go.Pie(
        labels=["Toss winner also won", "Toss winner lost"],
        values=[toss_won, total - toss_won],
        hole=0.45,
        marker_colors=["#00CC96", "#EF553B"],
        textinfo="label+percent",
    )
)
fig.update_layout(title="Toss Winner = Match Winner?", template=TEMPLATE)
show(fig)

# %% [markdown]
# ### 2d. Venues — most matches hosted

# %%
venue_counts = matches["venue"].value_counts().reset_index()
venue_counts.columns = ["venue", "matches"]

fig = px.bar(
    venue_counts.head(20),
    x="matches",
    y="venue",
    orientation="h",
    title="Top 20 Venues by Matches Hosted",
    labels={"matches": "Matches", "venue": "Venue"},
    color="matches",
    color_continuous_scale="Plasma",
    template=TEMPLATE,
)
fig.update_layout(yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False)
show(fig)

# %% [markdown]
# ### 2e. Win margins — biggest victories

# %%
top_run_wins = matches.nlargest(10, "win_by_runs")[
    ["season", "team1", "team2", "winner", "win_by_runs", "venue"]
]
print("Top 10 wins by runs margin:")
print(top_run_wins.to_string(index=False))

top_wicket_wins = matches.nlargest(10, "win_by_wickets")[
    ["season", "team1", "team2", "winner", "win_by_wickets", "venue"]
]
print("\nTop 10 wins by wickets margin:")
print(top_wicket_wins.to_string(index=False))

# %% [markdown]
# ### 2f. Player of the Match — top 15

# %%
potm = matches["player_of_match"].value_counts().head(15).reset_index()
potm.columns = ["player", "awards"]

fig = px.bar(
    potm,
    x="awards",
    y="player",
    orientation="h",
    title="Top 15 Player of the Match Award Winners",
    color="awards",
    color_continuous_scale="Turbo",
    template=TEMPLATE,
)
fig.update_layout(yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False)
show(fig)

# %% [markdown]
# ### 2g. Team win counts

# %%
wins = matches.dropna(subset=["winner"])["winner"].value_counts().reset_index()
wins.columns = ["team", "wins"]

fig = px.bar(
    wins,
    x="team",
    y="wins",
    title="Total Wins per Team (All Seasons)",
    color="wins",
    color_continuous_scale="Cividis",
    template=TEMPLATE,
)
fig.update_layout(coloraxis_showscale=False)
show(fig)

# %% [markdown]
# ## 3. Deliveries-Based Batting Analytics

# %% [markdown]
# ### 3a. Top 15 run scorers

# %%
bat_stats = batting_career_stats(deliveries)
top_batsmen = bat_stats[bat_stats["balls_faced"] >= 200].head(15)

fig = px.scatter(
    bat_stats[bat_stats["balls_faced"] >= 300],
    x="average",
    y="strike_rate",
    size="runs",
    color="boundary_pct",
    hover_name="batsman",
    hover_data={"runs": True, "matches": True},
    title="Batsman Efficiency: Average vs Strike Rate<br>(bubble = runs, color = boundary %)",
    labels={"average": "Batting Average", "strike_rate": "Strike Rate"},
    color_continuous_scale="RdYlGn",
    template=TEMPLATE,
)
fig.add_hline(
    y=bat_stats[bat_stats["balls_faced"] >= 300]["strike_rate"].median(),
    line_dash="dash",
    line_color="white",
    opacity=0.4,
    annotation_text="Median SR",
)
fig.add_vline(
    x=bat_stats[bat_stats["balls_faced"] >= 300]["average"].median(),
    line_dash="dash",
    line_color="white",
    opacity=0.4,
    annotation_text="Median Avg",
)
show(fig)

# %%
fig = px.bar(
    top_batsmen.sort_values("runs"),
    x="runs",
    y="batsman",
    orientation="h",
    title="Top 15 Run Scorers (min 200 balls faced)",
    color="strike_rate",
    color_continuous_scale="Plasma",
    hover_data=["matches", "average", "strike_rate", "boundary_pct"],
    template=TEMPLATE,
)
fig.update_layout(coloraxis_colorbar_title="SR")
show(fig)

# %% [markdown]
# ### 3b. Phase-wise strike rate (Powerplay / Middle / Death)

# %%
phase_bat = batting_phase_stats(deliveries)
phase_bat = phase_bat[phase_bat["balls"] >= 30]

top_phase_players = phase_bat.groupby("batsman")["balls"].sum().nlargest(20).index
phase_bat_top = phase_bat[phase_bat["batsman"].isin(top_phase_players)]

fig = px.bar(
    phase_bat_top,
    x="batsman",
    y="strike_rate",
    color="phase",
    barmode="group",
    title="Phase-wise Strike Rate — Top 20 Batsmen (min 30 balls per phase)",
    labels={"strike_rate": "Strike Rate", "phase": "Phase"},
    color_discrete_map={
        "Powerplay": "#00CC96",
        "Middle": "#FFA15A",
        "Death": "#EF553B",
    },
    template=TEMPLATE,
)
fig.update_xaxes(tickangle=45)
show(fig)

# %% [markdown]
# ### 3c. Dot ball pressure index

# %%
pressure = bat_stats[bat_stats["balls_faced"] >= 200].nlargest(15, "dot_ball_pct")

fig = px.bar(
    pressure.sort_values("dot_ball_pct"),
    x="dot_ball_pct",
    y="batsman",
    orientation="h",
    title="Top 15 Batsmen by Dot Ball % (min 200 balls)",
    labels={"dot_ball_pct": "Dot Ball %"},
    color="dot_ball_pct",
    color_continuous_scale="Reds",
    template=TEMPLATE,
)
fig.update_layout(coloraxis_showscale=False)
show(fig)

# %% [markdown]
# ## 4. Deliveries-Based Bowling Analytics

# %% [markdown]
# ### 4a. Top wicket takers

# %%
bowl_stats = bowling_career_stats(deliveries)
top_bowlers = bowl_stats[bowl_stats["balls_bowled"] >= 300].head(15)

fig = px.scatter(
    bowl_stats[bowl_stats["balls_bowled"] >= 300],
    x="economy",
    y="strike_rate",
    size="wickets",
    color="dot_ball_pct",
    hover_name="bowler",
    hover_data={"wickets": True, "matches": True},
    title="Bowler Efficiency: Economy vs Strike Rate<br>(bubble = wickets, color = dot ball %)",
    labels={"economy": "Economy Rate", "strike_rate": "Bowling SR"},
    color_continuous_scale="RdYlGn_r",
    template=TEMPLATE,
)
fig.add_hline(
    y=bowl_stats[bowl_stats["balls_bowled"] >= 300]["strike_rate"].median(),
    line_dash="dash",
    line_color="white",
    opacity=0.4,
)
fig.add_vline(
    x=bowl_stats[bowl_stats["balls_bowled"] >= 300]["economy"].median(),
    line_dash="dash",
    line_color="white",
    opacity=0.4,
)
show(fig)

# %%
fig = px.bar(
    top_bowlers.sort_values("wickets"),
    x="wickets",
    y="bowler",
    orientation="h",
    title="Top 15 Wicket Takers (min 300 balls bowled)",
    color="economy",
    color_continuous_scale="RdYlGn_r",
    hover_data=["matches", "average", "economy", "dot_ball_pct"],
    template=TEMPLATE,
)
fig.update_layout(coloraxis_colorbar_title="Economy")
show(fig)

# %% [markdown]
# ### 4b. Phase-wise bowling economy

# %%
phase_bowl = bowling_phase_economy(deliveries)
phase_bowl = phase_bowl[phase_bowl["balls"] >= 24]

top_bowl_players = phase_bowl.groupby("bowler")["balls"].sum().nlargest(20).index
phase_bowl_top = phase_bowl[phase_bowl["bowler"].isin(top_bowl_players)]

fig = px.bar(
    phase_bowl_top,
    x="bowler",
    y="economy",
    color="phase",
    barmode="group",
    title="Phase-wise Economy Rate — Top 20 Bowlers (min 24 balls per phase)",
    color_discrete_map={
        "Powerplay": "#00CC96",
        "Middle": "#FFA15A",
        "Death": "#EF553B",
    },
    template=TEMPLATE,
)
fig.update_xaxes(tickangle=45)
show(fig)

# %% [markdown]
# ## 5. Venue Analytics

# %% [markdown]
# ### 5a. Average first innings score by venue

# %%
venue_scores = venue_par_scores(deliveries, matches)

fig = px.bar(
    venue_scores[venue_scores["matches_hosted"] >= 5].sort_values("avg_first_innings"),
    x="avg_first_innings",
    y="venue",
    orientation="h",
    title="Average 1st Innings Score by Venue (min 5 matches)",
    color="avg_first_innings",
    color_continuous_scale="Blues",
    hover_data=["median_first_innings", "matches_hosted"],
    template=TEMPLATE,
)
fig.update_layout(yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False)
show(fig)

# %% [markdown]
# ## 6. Head-to-Head Matrix

# %%
h2h = head_to_head_matrix(matches)

fig = px.imshow(
    h2h,
    title="Head-to-Head Win Count Matrix",
    color_continuous_scale="Blues",
    aspect="auto",
    template=TEMPLATE,
    text_auto=True,
)
fig.update_layout(
    xaxis_title="Opponent",
    yaxis_title="Winner",
    coloraxis_showscale=False,
)
show(fig)

# %% [markdown]
# ## 7. Super Over Analysis

# %%
so_deliveries = deliveries[deliveries["is_super_over"]]
so_matches = so_deliveries["match_id"].nunique()
print(f"Super over matches in dataset: {so_matches}")

if so_matches > 0:
    so_scores = (
        so_deliveries.groupby(["match_id", "batting_team"])
        .agg(runs=("total_runs", "sum"), wickets=("is_wicket", "sum"))
        .reset_index()
    )
    so_merged = so_scores.merge(
        matches[["id", "winner"]].rename(columns={"id": "match_id"}),
        on="match_id",
    )
    so_merged["won"] = so_merged["batting_team"] == so_merged["winner"]

    fig = px.scatter(
        so_merged,
        x="runs",
        y="wickets",
        color="won",
        symbol="batting_team",
        title="Super Over Results: Runs vs Wickets",
        color_discrete_map={True: "#00CC96", False: "#EF553B"},
        template=TEMPLATE,
    )
    show(fig)
else:
    print("No super over data in this dataset.")

# %% [markdown]
# ## 8. Season-over-Season Batting Trend

# %%
bat_season = (
    deliveries[~deliveries["is_super_over"]]
    .merge(matches[["id", "season"]], left_on="match_id", right_on="id", how="left")
    .groupby("season")
    .agg(
        total_runs=("total_runs", "sum"),
        total_balls=("is_legal_delivery", "sum"),
        total_sixes=("batsman_runs", lambda x: (x == 6).sum()),
        total_fours=("batsman_runs", lambda x: (x == 4).sum()),
    )
    .reset_index()
)
bat_season["avg_run_rate"] = (
    bat_season["total_runs"] / (bat_season["total_balls"] / 6)
).round(2)
bat_season["six_rate"] = (
    bat_season["total_sixes"] / bat_season["total_balls"] * 6
).round(3)

fig = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=("Average Run Rate per Season", "Six Rate per Season"),
)

fig.add_trace(
    go.Scatter(
        x=bat_season["season"],
        y=bat_season["avg_run_rate"],
        mode="lines+markers",
        name="Run Rate",
        line=dict(color="#00CC96"),
    ),
    row=1,
    col=1,
)

fig.add_trace(
    go.Scatter(
        x=bat_season["season"],
        y=bat_season["six_rate"],
        mode="lines+markers",
        name="Six Rate",
        line=dict(color="#EF553B"),
    ),
    row=1,
    col=2,
)

fig.update_layout(
    title="Has T20 Batting Evolved Over Seasons?",
    template=TEMPLATE,
    showlegend=False,
)
show(fig)
