"""
features.py
-----------
Computes cricket-specific features from cleaned matches + deliveries DataFrames.
All functions are pure (no side effects on the input DataFrames).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Batting analytics
# ---------------------------------------------------------------------------


def batting_career_stats(deliveries: pd.DataFrame) -> pd.DataFrame:
    """
    Per-player career batting stats derived from deliveries.

    Returns columns:
        batsman, matches, innings, runs, balls_faced, fours, sixes,
        strike_rate, average, dot_ball_pct, boundary_pct
    """
    legal = deliveries[~deliveries["is_super_over"]].copy()

    runs_per_ball = (
        legal.groupby(["match_id", "batsman"])["batsman_runs"]
        .sum()
        .reset_index(name="match_runs")
    )

    # Innings count (a batsman can appear in multiple matches)
    innings = legal.groupby("batsman")["match_id"].nunique().reset_index(name="matches")

    # Aggregate
    agg = (
        legal.groupby("batsman")
        .agg(
            runs=("batsman_runs", "sum"),
            balls_faced=("is_legal_delivery", "sum"),
            fours=("batsman_runs", lambda x: (x == 4).sum()),
            sixes=("batsman_runs", lambda x: (x == 6).sum()),
            dismissals=("is_wicket", "sum"),
            dot_balls=("batsman_runs", lambda x: (x == 0).sum()),
        )
        .reset_index()
    )

    agg = agg.merge(innings, on="batsman")
    agg["strike_rate"] = (
        agg["runs"] / agg["balls_faced"].replace(0, np.nan) * 100
    ).round(2)
    agg["average"] = (agg["runs"] / agg["dismissals"].replace(0, np.nan)).round(2)
    agg["dot_ball_pct"] = (
        agg["dot_balls"] / agg["balls_faced"].replace(0, np.nan) * 100
    ).round(2)
    agg["boundary_runs"] = agg["fours"] * 4 + agg["sixes"] * 6
    agg["boundary_pct"] = (
        agg["boundary_runs"] / agg["runs"].replace(0, np.nan) * 100
    ).round(2)

    return agg.sort_values("runs", ascending=False).reset_index(drop=True)


def batting_phase_stats(deliveries: pd.DataFrame) -> pd.DataFrame:
    """
    Phase-wise (Powerplay / Middle / Death) batting stats per batsman.
    """
    df = deliveries[~deliveries["is_super_over"]].copy()
    df["phase"] = pd.cut(
        df["over"],
        bins=[0, 6, 15, 20],
        labels=["Powerplay", "Middle", "Death"],
        right=True,
    )

    agg = (
        df.groupby(["batsman", "phase"], observed=False)
        .agg(
            runs=("batsman_runs", "sum"),
            balls=("is_legal_delivery", "sum"),
            sixes=("batsman_runs", lambda x: (x == 6).sum()),
        )
        .reset_index()
    )

    agg["phase"] = agg["phase"].astype(str)
    agg["strike_rate"] = (agg["runs"] / agg["balls"].replace(0, np.nan) * 100).round(2)
    return agg


# ---------------------------------------------------------------------------
# Bowling analytics
# ---------------------------------------------------------------------------


def bowling_career_stats(deliveries: pd.DataFrame) -> pd.DataFrame:
    """
    Per-bowler career bowling stats.

    Returns columns:
        bowler, matches, wickets, runs_conceded, overs_bowled,
        economy, average, strike_rate, dot_ball_pct
    """
    legal = deliveries[
        ~deliveries["is_super_over"] & deliveries["is_legal_delivery"]
    ].copy()

    matches = (
        deliveries.groupby("bowler")["match_id"].nunique().reset_index(name="matches")
    )

    agg = (
        legal.groupby("bowler")
        .agg(
            wickets=("is_wicket", "sum"),
            runs_conceded=("total_runs", "sum"),
            balls_bowled=("is_legal_delivery", "sum"),
            dot_balls=("total_runs", lambda x: (x == 0).sum()),
        )
        .reset_index()
    )

    agg = agg.merge(matches, on="bowler")
    agg["overs_bowled"] = (agg["balls_bowled"] / 6).round(2)
    agg["economy"] = (
        agg["runs_conceded"] / agg["overs_bowled"].replace(0, np.nan)
    ).round(2)
    agg["average"] = (agg["runs_conceded"] / agg["wickets"].replace(0, np.nan)).round(2)
    agg["strike_rate"] = (
        agg["balls_bowled"] / agg["wickets"].replace(0, np.nan)
    ).round(2)
    agg["dot_ball_pct"] = (
        agg["dot_balls"] / agg["balls_bowled"].replace(0, np.nan) * 100
    ).round(2)

    return agg.sort_values("wickets", ascending=False).reset_index(drop=True)


def bowling_phase_economy(deliveries: pd.DataFrame) -> pd.DataFrame:
    """Economy rate broken down by phase for each bowler."""
    df = deliveries[
        ~deliveries["is_super_over"] & deliveries["is_legal_delivery"]
    ].copy()
    df["phase"] = pd.cut(
        df["over"],
        bins=[0, 6, 15, 20],
        labels=["Powerplay", "Middle", "Death"],
        right=True,
    )

    agg = (
        df.groupby(["bowler", "phase"], observed=False)
        .agg(
            runs=("total_runs", "sum"),
            balls=("is_legal_delivery", "sum"),
            wickets=("is_wicket", "sum"),
        )
        .reset_index()
    )

    agg["phase"] = agg["phase"].astype(str)
    agg["economy"] = (agg["runs"] / (agg["balls"] / 6).replace(0, np.nan)).round(2)
    return agg


# ---------------------------------------------------------------------------
# Partnership analytics
# ---------------------------------------------------------------------------


def partnership_stats(deliveries: pd.DataFrame) -> pd.DataFrame:
    """
    Compute partnership runs for each (match, inning, batsman, non_striker) group.
    Requires 'non_striker' column in deliveries.
    """
    if "non_striker" not in deliveries.columns:
        raise ValueError(
            "deliveries must contain 'non_striker' column for partnership analysis"
        )

    df = deliveries[~deliveries["is_super_over"]].copy()

    # Normalize pair order so (A,B) and (B,A) are the same
    df["pair"] = df.apply(
        lambda r: tuple(sorted([r["batsman"], r["non_striker"]])), axis=1
    )

    agg = (
        df.groupby(["match_id", "inning", "pair"])
        .agg(
            runs=("batsman_runs", "sum"),
            balls=("is_legal_delivery", "sum"),
        )
        .reset_index()
    )

    agg[["player1", "player2"]] = pd.DataFrame(agg["pair"].tolist(), index=agg.index)
    agg.drop(columns=["pair"], inplace=True)

    summary = (
        agg.groupby(["player1", "player2"])
        .agg(
            total_runs=("runs", "sum"),
            partnerships=("runs", "count"),
            avg_partnership=("runs", "mean"),
        )
        .reset_index()
        .sort_values("total_runs", ascending=False)
    )

    summary["avg_partnership"] = summary["avg_partnership"].round(1)
    return summary.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Venue analytics
# ---------------------------------------------------------------------------


def venue_par_scores(deliveries: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """
    Compute average 1st innings score and average chase target per venue.
    """
    first_innings = deliveries[
        (deliveries["inning"] == 1) & (~deliveries["is_super_over"])
    ]

    score_per_match = (
        first_innings.groupby("match_id")["total_runs"]
        .sum()
        .reset_index(name="first_innings_score")
    )

    merged = score_per_match.merge(
        matches[["id", "venue", "winner", "team1", "team2"]],
        left_on="match_id",
        right_on="id",
        how="left",
    )

    venue_stats = (
        merged.groupby("venue")
        .agg(
            avg_first_innings=("first_innings_score", "mean"),
            median_first_innings=("first_innings_score", "median"),
            matches_hosted=("match_id", "count"),
        )
        .reset_index()
        .sort_values("avg_first_innings", ascending=False)
    )

    venue_stats["avg_first_innings"] = venue_stats["avg_first_innings"].round(1)
    venue_stats["median_first_innings"] = venue_stats["median_first_innings"].round(1)
    return venue_stats


# ---------------------------------------------------------------------------
# Head-to-head win matrix
# ---------------------------------------------------------------------------


def head_to_head_matrix(matches: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a pivot table: rows = team, cols = opponent, values = win count.
    """
    rows = []
    for _, row in matches.dropna(subset=["winner"]).iterrows():
        rows.append(
            {
                "team": row["winner"],
                "opponent": (
                    row["team1"] if row["winner"] == row["team2"] else row["team2"]
                ),
            }
        )

    df = pd.DataFrame(rows)
    matrix = df.groupby(["team", "opponent"]).size().unstack(fill_value=0)
    return matrix


# ---------------------------------------------------------------------------
# ML feature engineering
# ---------------------------------------------------------------------------


def build_match_features(matches: pd.DataFrame) -> pd.DataFrame:
    """
    Build tabular feature matrix for match-outcome prediction.

    Features: toss_winner_is_team1, toss_decision_encoded, venue_encoded,
              team1_encoded, team2_encoded, season
    Target:   team1_won (1 if team1 wins, 0 if team2 wins)
    """
    df = matches.dropna(subset=["winner"]).copy()

    df["toss_winner_is_team1"] = (df["toss_winner"] == df["team1"]).astype(int)
    df["bat_first"] = (df["toss_decision"] == "bat").astype(int)
    df["team1_won"] = (df["winner"] == df["team1"]).astype(int)

    for col in ["venue", "team1", "team2"]:
        df[f"{col}_enc"] = df[col].astype("category").cat.codes

    features = [
        "toss_winner_is_team1",
        "bat_first",
        "venue_enc",
        "team1_enc",
        "team2_enc",
        "season",
    ]
    return df[features + ["team1_won"]].dropna()


def build_win_probability_features(
    deliveries: pd.DataFrame,
    matches: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a ball-by-ball feature matrix for live win probability modelling.

    Each row = state of the game at end of each over in the 2nd innings.
    Features: runs_scored, wickets_fallen, current_rr, required_rr,
              balls_remaining, runs_required, venue_enc, batting_team_enc
    Target:   batting_team_won
    """
    # 1st innings total per match
    first_inn = (
        deliveries[deliveries["inning"] == 1]
        .groupby("match_id")["total_runs"]
        .sum()
        .reset_index(name="target")
    )
    first_inn["target"] += 1  # team needs target+1 to win

    # 2nd innings cumulative over-by-over
    second_inn = deliveries[
        (deliveries["inning"] == 2) & (~deliveries["is_super_over"])
    ].copy()

    over_agg = (
        second_inn.groupby(["match_id", "over"])
        .agg(
            runs_in_over=("total_runs", "sum"),
            wickets_in_over=("is_wicket", "sum"),
            batting_team=("batting_team", "first"),
        )
        .reset_index()
    )

    over_agg = over_agg.sort_values(["match_id", "over"])
    over_agg["runs_scored"] = over_agg.groupby("match_id")["runs_in_over"].cumsum()
    over_agg["wickets_fallen"] = over_agg.groupby("match_id")[
        "wickets_in_over"
    ].cumsum()
    over_agg["balls_completed"] = (over_agg["over"]) * 6
    over_agg["balls_remaining"] = 120 - over_agg["balls_completed"]

    over_agg = over_agg.merge(first_inn, on="match_id", how="left")
    over_agg["runs_required"] = over_agg["target"] - over_agg["runs_scored"]
    over_agg["current_rr"] = (
        over_agg["runs_scored"] / (over_agg["balls_completed"] / 6).replace(0, np.nan)
    ).round(2)
    over_agg["required_rr"] = (
        over_agg["runs_required"] / (over_agg["balls_remaining"] / 6).replace(0, np.nan)
    ).round(2)

    # Join winner from matches
    over_agg = over_agg.merge(
        matches[["id", "winner", "venue"]].rename(columns={"id": "match_id"}),
        on="match_id",
        how="left",
    )
    over_agg["batting_team_won"] = (
        over_agg["batting_team"] == over_agg["winner"]
    ).astype(int)

    # Encode categoricals
    over_agg["venue_enc"] = over_agg["venue"].astype("category").cat.codes
    over_agg["batting_team_enc"] = over_agg["batting_team"].astype("category").cat.codes

    features = [
        "over",
        "runs_scored",
        "wickets_fallen",
        "current_rr",
        "required_rr",
        "balls_remaining",
        "runs_required",
        "venue_enc",
        "batting_team_enc",
    ]
    df_out = over_agg[features + ["batting_team_won", "match_id"]].dropna()
    return df_out


def build_match_features_v2(matches: pd.DataFrame) -> pd.DataFrame:
    """
    Adds historical win-rate features per team (computed on rolling prior matches).
    Avoids leakage: win rate for match N uses only matches 0..N-1.
    """
    df = (
        matches.dropna(subset=["winner"])
        .copy()
        .sort_values("date")
        .reset_index(drop=True)
    )

    # Rolling win rate per team (computed before each match)
    team_wins = {}
    team_played = {}

    win_rate_team1 = []
    win_rate_team2 = []

    for _, row in df.iterrows():
        t1, t2, winner = row["team1"], row["team2"], row["winner"]

        # win rate before this match
        w1 = team_wins.get(t1, 0) / max(team_played.get(t1, 1), 1)
        w2 = team_wins.get(t2, 0) / max(team_played.get(t2, 1), 1)
        win_rate_team1.append(round(w1, 4))
        win_rate_team2.append(round(w2, 4))

        # update after
        team_played[t1] = team_played.get(t1, 0) + 1
        team_played[t2] = team_played.get(t2, 0) + 1
        team_wins[winner] = team_wins.get(winner, 0) + 1

    df["win_rate_team1"] = win_rate_team1
    df["win_rate_team2"] = win_rate_team2
    df["win_rate_diff"] = df["win_rate_team1"] - df["win_rate_team2"]
    df["toss_winner_is_team1"] = (df["toss_winner"] == df["team1"]).astype(int)
    df["bat_first"] = (df["toss_decision"] == "bat").astype(int)
    df["team1_won"] = (df["winner"] == df["team1"]).astype(int)

    for col in ["venue", "team1", "team2"]:
        df[f"{col}_enc"] = df[col].astype("category").cat.codes

    features = [
        "toss_winner_is_team1",
        "bat_first",
        "venue_enc",
        "team1_enc",
        "team2_enc",
        "season",
        "win_rate_team1",
        "win_rate_team2",
        "win_rate_diff",
    ]
    return df[features + ["team1_won"]].dropna()
