"""
rolling_features.py
-------------------
Computes rolling player-form features aggregated to team level for
match-winner prediction.

All computations are leak-free: rolling stats for match M use only
deliveries from matches whose date is strictly before match M's date.
This is enforced by shift(1) inside the rolling window.

Public API
----------
compute_team_rolling_form(deliveries, matches, n_matches=5)
    → DataFrame: (match_id, team, rolling_bat_avg, rolling_bat_sr,
                  rolling_bowl_econ, rolling_bowl_sr)
    One row per (match, team). Represents form *entering* that match.

build_match_features_v3(matches, deliveries, n_matches=5)
    → DataFrame: full match-winner feature matrix (v2 base + rolling player
                 form columns) with target column team1_won.
"""

from __future__ import annotations

import logging
from typing import Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Dismissal kinds that should not be credited to the bowler
_NON_BOWLER_DISMISSALS = {"run out", "retired hurt", "obstructing the field"}

# Default n_matches for rolling window
N_DEFAULT: int = 5


# ---------------------------------------------------------------------------
# Private helpers — per-(match_id, player) stats
# ---------------------------------------------------------------------------


def _player_match_batting(deliveries: pd.DataFrame) -> pd.DataFrame:
    """
    Per-(match_id, player) batting stats. Super-overs excluded.

    Returns columns: match_id, player, bat_avg, bat_sr
    bat_avg: runs/dismissals; if not out (dismissals=0), bat_avg = runs (cricket convention)
    bat_sr : runs/balls × 100; 0 if no balls faced
    """
    d = deliveries[~deliveries["is_super_over"]].copy()

    agg = (
        d.groupby(["match_id", "batsman"])
        .agg(
            runs=("batsman_runs", "sum"),
            balls=("is_legal_delivery", "sum"),
            dismissals=("is_wicket", "sum"),
        )
        .reset_index()
        .rename(columns={"batsman": "player"})
    )

    agg["bat_avg"] = np.where(
        agg["dismissals"] > 0,
        agg["runs"] / agg["dismissals"],
        agg["runs"].astype(float),  # not-out: avg = runs scored
    )
    agg["bat_sr"] = np.where(
        agg["balls"] > 0,
        agg["runs"] / agg["balls"] * 100.0,
        0.0,
    )
    return agg[["match_id", "player", "bat_avg", "bat_sr"]]


def _player_match_bowling(deliveries: pd.DataFrame) -> pd.DataFrame:
    """
    Per-(match_id, bowler) bowling stats. Legal deliveries only; super-overs excluded.

    Returns columns: match_id, player, bowl_econ, bowl_sr
    bowl_econ: runs conceded per over (byes + legbyes excluded)
    bowl_sr  : balls per wicket; if no wickets, value = balls bowled (penalises wicketless spells)
    """
    d = deliveries[
        ~deliveries["is_super_over"] & deliveries["is_legal_delivery"]
    ].copy()

    # Exclude non-bowler dismissals from wicket count
    if "dismissal_kind" in d.columns:
        d["_wkt"] = d["player_dismissed"].notna() & (
            ~d["dismissal_kind"].isin(_NON_BOWLER_DISMISSALS)
        )
    else:
        d["_wkt"] = d["player_dismissed"].notna()

    # Runs conceded: strip byes and legbyes
    bye = d["bye_runs"] if "bye_runs" in d.columns else pd.Series(0, index=d.index)
    leg = (
        d["legbye_runs"] if "legbye_runs" in d.columns else pd.Series(0, index=d.index)
    )
    d["_rc"] = d["total_runs"] - bye - leg

    agg = (
        d.groupby(["match_id", "bowler"])
        .agg(
            wickets=("_wkt", "sum"),
            runs_conceded=("_rc", "sum"),
            balls=("is_legal_delivery", "sum"),
        )
        .reset_index()
        .rename(columns={"bowler": "player"})
    )

    agg["bowl_econ"] = np.where(
        agg["balls"] > 0,
        agg["runs_conceded"] / (agg["balls"] / 6.0),
        0.0,
    )
    agg["bowl_sr"] = np.where(
        agg["wickets"] > 0,
        agg["balls"] / agg["wickets"].astype(float),
        agg["balls"].astype(float),  # wicketless: use balls as penalty
    )
    return agg[["match_id", "player", "bowl_econ", "bowl_sr"]]


def _rolling_shift(series: pd.Series, n: int) -> pd.Series:
    """
    Shift by 1 then rolling mean over n matches (min_periods=1).
    shift(1) ensures the current match's stats are excluded.
    """
    return series.shift(1).rolling(n, min_periods=1).mean()


# ---------------------------------------------------------------------------
# Public: team-level rolling form
# ---------------------------------------------------------------------------


def compute_team_rolling_form(
    deliveries: pd.DataFrame,
    matches: pd.DataFrame,
    n_matches: int = N_DEFAULT,
) -> pd.DataFrame:
    """
    Rolling n-match batting and bowling form per (match_id, team).

    Returns
    -------
    DataFrame with columns:
        match_id, team,
        rolling_bat_avg, rolling_bat_sr,
        rolling_bowl_econ, rolling_bowl_sr

    Each row represents the team's form *entering* that match.
    Leak-free: current match is always excluded from the rolling window.
    """
    logger.info("Computing per-player batting stats…")
    bat = _player_match_batting(deliveries)

    logger.info("Computing per-player bowling stats…")
    bowl = _player_match_bowling(deliveries)

    # Attach dates so we can sort chronologically
    match_dates = matches[["id", "date"]].rename(columns={"id": "match_id"})
    bat = bat.merge(match_dates, on="match_id", how="left")
    bowl = bowl.merge(match_dates, on="match_id", how="left")

    # Sort per player by date (then match_id as tiebreaker)
    bat = bat.sort_values(["player", "date", "match_id"]).copy()
    bowl = bowl.sort_values(["player", "date", "match_id"]).copy()

    # Rolling per player — shift(1) excludes current match
    for col in ["bat_avg", "bat_sr"]:
        bat[f"rolling_{col}"] = bat.groupby("player")[col].transform(
            lambda x: _rolling_shift(x, n_matches)
        )

    for col in ["bowl_econ", "bowl_sr"]:
        bowl[f"rolling_{col}"] = bowl.groupby("player")[col].transform(
            lambda x: _rolling_shift(x, n_matches)
        )

    # Cold-start fill: player's first match — no prior data — use global prior
    bat["rolling_bat_avg"] = bat["rolling_bat_avg"].fillna(bat["bat_avg"].mean())
    bat["rolling_bat_sr"] = bat["rolling_bat_sr"].fillna(bat["bat_sr"].mean())
    bowl["rolling_bowl_econ"] = bowl["rolling_bowl_econ"].fillna(
        bowl["bowl_econ"].mean()
    )
    bowl["rolling_bowl_sr"] = bowl["rolling_bowl_sr"].fillna(bowl["bowl_sr"].median())

    # Map players to their teams using deliveries
    # batting_team  = batsman's team; bowling_team = bowler's team
    bat_team = (
        deliveries[~deliveries["is_super_over"]]
        .groupby(["match_id", "batsman"])["batting_team"]
        .first()
        .reset_index()
        .rename(columns={"batsman": "player", "batting_team": "team"})
    )
    bowl_team = (
        deliveries[~deliveries["is_super_over"]]
        .groupby(["match_id", "bowler"])["bowling_team"]
        .first()
        .reset_index()
        .rename(columns={"bowler": "player", "bowling_team": "team"})
    )

    bat = bat.merge(bat_team, on=["match_id", "player"], how="left")
    bowl = bowl.merge(bowl_team, on=["match_id", "player"], how="left")

    # Aggregate to team level: mean rolling stat across all players
    team_bat = (
        bat.dropna(subset=["team"])
        .groupby(["match_id", "team"])
        .agg(
            rolling_bat_avg=("rolling_bat_avg", "mean"),
            rolling_bat_sr=("rolling_bat_sr", "mean"),
        )
        .reset_index()
    )

    team_bowl = (
        bowl.dropna(subset=["team"])
        .groupby(["match_id", "team"])
        .agg(
            rolling_bowl_econ=("rolling_bowl_econ", "mean"),
            rolling_bowl_sr=("rolling_bowl_sr", "mean"),
        )
        .reset_index()
    )

    team_form = team_bat.merge(team_bowl, on=["match_id", "team"], how="outer")

    # Round for storage efficiency
    for col in [
        "rolling_bat_avg",
        "rolling_bat_sr",
        "rolling_bowl_econ",
        "rolling_bowl_sr",
    ]:
        if col in team_form.columns:
            team_form[col] = team_form[col].round(3)

    result = team_form.dropna()
    logger.info(
        "compute_team_rolling_form: %d rows, %d teams",
        len(result),
        result["team"].nunique(),
    )
    return result


# ---------------------------------------------------------------------------
# Public: enhanced match feature matrix
# ---------------------------------------------------------------------------


def build_match_features_v3(
    matches: pd.DataFrame,
    deliveries: pd.DataFrame,
    n_matches: int = N_DEFAULT,
) -> pd.DataFrame:
    """
    Full match-winner feature matrix: v2 base + rolling player-form columns.

    New features vs v2:
        team1_rolling_bat_avg   — mean rolling batting avg of team1's players
        team1_rolling_bat_sr    — mean rolling strike rate of team1's players
        team2_rolling_bat_avg
        team2_rolling_bat_sr
        team1_rolling_bowl_econ — mean rolling economy of team1's bowlers
        team1_rolling_bowl_sr   — mean rolling bowling SR of team1's bowlers
        team2_rolling_bowl_econ
        team2_rolling_bowl_sr
        rolling_bat_avg_diff    — team1 − team2 (positive = team1 stronger)
        rolling_bat_sr_diff     — team1 − team2
        rolling_bowl_econ_diff  — team1 − team2 (negative = team1 more economical)

    Total features: 9 (v2) + 11 (rolling) = 20 features.
    Target column: team1_won.
    """
    # ── 1. Build base dataframe (same logic as build_match_features_v2) ───
    df = (
        matches.dropna(subset=["winner"])
        .copy()
        .sort_values("date")
        .reset_index(drop=True)
    )
    df["match_id"] = df["id"].astype(int)

    # Rolling win rates (leak-free: state before each match)
    team_wins: Dict[str, int] = {}
    team_played: Dict[str, int] = {}
    wr1_list, wr2_list = [], []

    for _, row in df.iterrows():
        t1, t2, winner = row["team1"], row["team2"], row["winner"]
        w1 = team_wins.get(t1, 0) / max(team_played.get(t1, 1), 1)
        w2 = team_wins.get(t2, 0) / max(team_played.get(t2, 1), 1)
        wr1_list.append(round(w1, 4))
        wr2_list.append(round(w2, 4))
        team_played[t1] = team_played.get(t1, 0) + 1
        team_played[t2] = team_played.get(t2, 0) + 1
        team_wins[winner] = team_wins.get(winner, 0) + 1

    df["win_rate_team1"] = wr1_list
    df["win_rate_team2"] = wr2_list
    df["win_rate_diff"] = df["win_rate_team1"] - df["win_rate_team2"]
    df["toss_winner_is_team1"] = (df["toss_winner"] == df["team1"]).astype(int)
    df["bat_first"] = (df["toss_decision"] == "bat").astype(int)
    df["team1_won"] = (df["winner"] == df["team1"]).astype(int)

    for col in ["venue", "team1", "team2"]:
        df[f"{col}_enc"] = df[col].astype("category").cat.codes

    # ── 2. Compute rolling team form ──────────────────────────────────────
    team_form = compute_team_rolling_form(deliveries, matches, n_matches)

    # Join rolling stats for team1 side
    t1_form = team_form.rename(
        columns={
            "team": "team1",
            "rolling_bat_avg": "team1_rolling_bat_avg",
            "rolling_bat_sr": "team1_rolling_bat_sr",
            "rolling_bowl_econ": "team1_rolling_bowl_econ",
            "rolling_bowl_sr": "team1_rolling_bowl_sr",
        }
    )
    # Join rolling stats for team2 side
    t2_form = team_form.rename(
        columns={
            "team": "team2",
            "rolling_bat_avg": "team2_rolling_bat_avg",
            "rolling_bat_sr": "team2_rolling_bat_sr",
            "rolling_bowl_econ": "team2_rolling_bowl_econ",
            "rolling_bowl_sr": "team2_rolling_bowl_sr",
        }
    )

    df = df.merge(t1_form, on=["match_id", "team1"], how="left")
    df = df.merge(t2_form, on=["match_id", "team2"], how="left")

    # Differential features (team1 − team2; model learns sign automatically)
    df["rolling_bat_avg_diff"] = (
        df["team1_rolling_bat_avg"] - df["team2_rolling_bat_avg"]
    )
    df["rolling_bat_sr_diff"] = df["team1_rolling_bat_sr"] - df["team2_rolling_bat_sr"]
    df["rolling_bowl_econ_diff"] = (
        df["team1_rolling_bowl_econ"] - df["team2_rolling_bowl_econ"]
    )

    # ── 3. Select and return ──────────────────────────────────────────────
    features = [
        # V2 base (9 features)
        "toss_winner_is_team1",
        "bat_first",
        "venue_enc",
        "team1_enc",
        "team2_enc",
        "season",
        "win_rate_team1",
        "win_rate_team2",
        "win_rate_diff",
        # Rolling batting (4 features)
        "team1_rolling_bat_avg",
        "team1_rolling_bat_sr",
        "team2_rolling_bat_avg",
        "team2_rolling_bat_sr",
        # Rolling bowling (4 features)
        "team1_rolling_bowl_econ",
        "team1_rolling_bowl_sr",
        "team2_rolling_bowl_econ",
        "team2_rolling_bowl_sr",
        # Differentials (3 features)
        "rolling_bat_avg_diff",
        "rolling_bat_sr_diff",
        "rolling_bowl_econ_diff",
    ]
    result = df[features + ["team1_won"]].dropna()
    logger.info(
        "build_match_features_v3: %d rows, %d features",
        len(result),
        len(features),
    )
    return result
