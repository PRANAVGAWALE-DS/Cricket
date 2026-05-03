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
    # super-over rows excluded for all career stats
    legal = deliveries[~deliveries["is_super_over"]].copy()

    # Legal-delivery frame — required so dot-ball numerator matches the
    # balls_faced denominator (both must exclude wides and no-balls).
    legal_del = legal[legal["is_legal_delivery"]].copy()

    # Match count per batsman (unique match appearances, super-overs excluded)
    match_counts = (
        legal.groupby("batsman")["match_id"].nunique().reset_index(name="matches")
    )

    # Aggregate — dot_balls computed on legal_del to avoid counting wides
    agg = (
        legal.groupby("batsman")
        .agg(
            runs=("batsman_runs", "sum"),
            balls_faced=("is_legal_delivery", "sum"),
            fours=("batsman_runs", lambda x: (x == 4).sum()),
            sixes=("batsman_runs", lambda x: (x == 6).sum()),
            dismissals=("is_wicket", "sum"),
        )
        .reset_index()
    )

    dot_balls = (
        legal_del.groupby("batsman")["batsman_runs"]
        .apply(lambda x: (x == 0).sum())
        .reset_index(name="dot_balls")
    )
    agg = agg.merge(dot_balls, on="batsman", how="left").fillna({"dot_balls": 0})
    agg["dot_balls"] = agg["dot_balls"].astype(int)

    agg = agg.merge(match_counts, on="batsman")
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

    # Match count derived from the same filtered frame as all other stats
    matches = (
        legal.groupby("bowler")["match_id"].nunique().reset_index(name="matches")
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

    # Normalize pair order so (A, B) and (B, A) map to the same key.
    # np.sort on the two-column array is ~20-40× faster than row-wise apply.
    sorted_pairs = np.sort(df[["batsman", "non_striker"]].values, axis=1)
    df["player1"] = sorted_pairs[:, 0]
    df["player2"] = sorted_pairs[:, 1]

    agg = (
        df.groupby(["match_id", "inning", "player1", "player2"])
        .agg(
            runs=("batsman_runs", "sum"),
            balls=("is_legal_delivery", "sum"),
        )
        .reset_index()
    )

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


# ---------------------------------------------------------------------------
# Score prediction features  (moved from models.py — feature builders
# belong in features.py, not in the model-training module)
# ---------------------------------------------------------------------------


def build_score_features(
    deliveries: pd.DataFrame, matches: pd.DataFrame
) -> pd.DataFrame:
    """
    Builds over-10 (halfway point) snapshot features to predict final 1st innings score.

    Prediction point: end of over 10 — teams make this calculation at the drinks break.
    Shifting from over-6 to over-10 gives the model 2/3 of the innings as context,
    substantially reducing irreducible uncertainty.

    Features:
        runs_10, wickets_10, current_rr, projected_score,
        scoring_pressure, boundaries_10,
        batting_team_enc, venue_enc, season
    Target: final_score

    NOTE: assumes 1-indexed overs (1–20) as validated by load_deliveries.
    """
    is_first = (deliveries["inning"] == 1) & (~deliveries["is_super_over"].astype(bool))

    half = (
        deliveries[is_first & (deliveries["over"] <= 10)]
        .groupby("match_id")
        .agg(
            runs_10=("total_runs", "sum"),
            wickets_10=("is_wicket", "sum"),
            boundaries_10=("batsman_runs", lambda x: ((x == 4) | (x == 6)).sum()),
            batting_team=("batting_team", "first"),
        )
        .reset_index()
    )
    half["current_rr"] = (half["runs_10"] / 10).round(3)
    half["projected_score"] = (half["current_rr"] * 20).round(1)

    full = (
        deliveries[is_first]
        .groupby("match_id")["total_runs"]
        .sum()
        .reset_index(name="final_score")
    )

    df = (
        half.merge(full, on="match_id")
        .merge(
            matches[["id", "venue", "season", "date"]].rename(
                columns={"id": "match_id"}
            ),
            on="match_id",
            how="left",
        )
        .sort_values("date")
        .reset_index(drop=True)
    )

    # Rolling venue avg run rate — no leakage, uses only prior matches
    venue_rr: dict = {}
    venue_rr_col = []
    for _, row in df.iterrows():
        v = row["venue"]
        prior = venue_rr.get(v, [])
        avg = round(sum(prior) / len(prior), 3) if prior else np.nan
        venue_rr_col.append(avg)
        prior.append(row["current_rr"])
        venue_rr[v] = prior

    global_rr = df["current_rr"].mean()
    df["venue_avg_rr"] = pd.Series(venue_rr_col).fillna(global_rr).values
    df["scoring_pressure"] = (df["current_rr"] - df["venue_avg_rr"]).round(3)

    df["batting_team_enc"] = df["batting_team"].astype("category").cat.codes
    df["venue_enc"] = df["venue"].astype("category").cat.codes

    features = [
        "runs_10",
        "wickets_10",
        "current_rr",
        "projected_score",
        "scoring_pressure",
        "boundaries_10",
        "batting_team_enc",
        "venue_enc",
        "season",
        "final_score",
    ]
    return df[features].dropna()


# ---------------------------------------------------------------------------
# POTM prediction features  (moved from models.py)
# ---------------------------------------------------------------------------


def build_potm_features(
    deliveries: pd.DataFrame, matches: pd.DataFrame
) -> pd.DataFrame:
    """
    Build player-level per-match features for POTM classification.

    For each (match_id, player) pair:
      - runs_scored, balls_faced, strike_rate
      - wickets_taken, runs_given, economy
      - player_won (1 if player's team won)

    Fix (C4): pure bowlers who never batted previously got player_won = 0
    regardless of match outcome because team lookup was batting-only.
    Now we build a unified player→team map from both batting and bowling
    sides (batting entry takes priority where both exist).

    Target: is_potm
    """
    batting = (
        deliveries[~deliveries["is_super_over"]]
        .groupby(["match_id", "batsman"])
        .agg(
            runs_scored=("batsman_runs", "sum"),
            balls_faced=("is_legal_delivery", "sum"),
        )
        .reset_index()
        .rename(columns={"batsman": "player"})
    )

    bowling = (
        deliveries[~deliveries["is_super_over"] & deliveries["is_legal_delivery"]]
        .groupby(["match_id", "bowler"])
        .agg(
            wickets_taken=("is_wicket", "sum"),
            runs_given=("total_runs", "sum"),
            balls_bowled=("is_legal_delivery", "sum"),
        )
        .reset_index()
        .rename(columns={"bowler": "player"})
    )

    # Full player list per match
    all_players = pd.concat(
        [
            batting[["match_id", "player"]],
            bowling[["match_id", "player"]],
        ]
    ).drop_duplicates()

    df = all_players.merge(batting, on=["match_id", "player"], how="left")
    df = df.merge(bowling, on=["match_id", "player"], how="left")
    df = df.fillna(0)

    df["strike_rate"] = (
        df["runs_scored"] / df["balls_faced"].replace(0, np.nan) * 100
    ).fillna(0)
    df["economy"] = (
        df["runs_given"] / (df["balls_bowled"] / 6).replace(0, np.nan)
    ).fillna(0)

    # POTM ground truth
    potm = matches[["id", "player_of_match", "winner"]].rename(
        columns={"id": "match_id", "player_of_match": "potm"}
    )
    df = df.merge(potm, on="match_id", how="left")
    df["is_potm"] = (df["player"] == df["potm"]).astype(int)

    # -----------------------------------------------------------------------
    # C4 fix: build a unified player→team lookup from BOTH batting and bowling
    # sides so that pure bowlers on the winning team are not always penalised
    # with player_won = 0.
    # bowling_team column in deliveries = the team currently bowling
    # = the bowler's own team.
    # batting_team column = the batsmen's team.
    # We union both, letting batting entries win on conflict (keep="last"
    # after concat places batting rows second so they overwrite).
    # -----------------------------------------------------------------------
    bowl_team = (
        deliveries[~deliveries["is_super_over"]]
        .groupby(["match_id", "bowler"])["bowling_team"]
        .first()
        .reset_index()
        .rename(columns={"bowler": "player", "bowling_team": "team"})
    )
    bat_team = (
        deliveries[~deliveries["is_super_over"]]
        .groupby(["match_id", "batsman"])["batting_team"]
        .first()
        .reset_index()
        .rename(columns={"batsman": "player", "batting_team": "team"})
    )
    # concat: bowl first, bat second → keep="last" retains bat entry on ties
    player_team = (
        pd.concat([bowl_team, bat_team])
        .drop_duplicates(subset=["match_id", "player"], keep="last")
    )

    df = df.merge(player_team, on=["match_id", "player"], how="left")
    df["player_won"] = (df["team"] == df["winner"]).astype(int)

    features = [
        "runs_scored",
        "balls_faced",
        "strike_rate",
        "wickets_taken",
        "runs_given",
        "economy",
        "player_won",
    ]
    return df[features + ["is_potm"]].dropna()