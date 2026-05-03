"""
data_loader.py
--------------
Loads, validates, and cleans the IPL matches + deliveries CSVs.

Expected schema
---------------
matches.csv  : id, season, city, date, team1, team2, toss_winner,
               toss_decision, result, dl_applied, winner, win_by_runs,
               win_by_wickets, player_of_match, venue, umpire1, umpire2,
               umpire3 (optional – dropped if present)

deliveries.csv: match_id, inning, batting_team, bowling_team, over, ball,
                batsman, non_striker, bowler, is_super_over, wide_runs,
                bye_runs, legbye_runs, noball_runs, penalty_runs,
                batsman_runs, extra_runs, total_runs, player_dismissed,
                dismissal_kind, fielder
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

TEAM_ABBREV: dict[str, str] = {
    "Sunrisers Hyderabad": "SRH",
    "Mumbai Indians": "MI",
    "Gujarat Lions": "GL",
    "Rising Pune Supergiant": "RPS",
    "Rising Pune Supergiants": "RPS",
    "Royal Challengers Bangalore": "RCB",
    "Kolkata Knight Riders": "KKR",
    "Delhi Daredevils": "DD",
    "Delhi Capitals": "DC",
    "Kings XI Punjab": "KXIP",
    "Punjab Kings": "PBKS",
    "Chennai Super Kings": "CSK",
    "Rajasthan Royals": "RR",
    "Deccan Chargers": "DC_old",
    "Kochi Tuskers Kerala": "KTK",
    "Pune Warriors": "PW",
}

MATCHES_REQUIRED_COLS = {
    "id",
    "season",
    "team1",
    "team2",
    "toss_winner",
    "toss_decision",
    "winner",
    "win_by_runs",
    "win_by_wickets",
    "player_of_match",
    "venue",
}

DELIVERIES_REQUIRED_COLS = {
    "match_id",
    "inning",
    "batting_team",
    "bowling_team",
    "over",
    "ball",
    "batsman",
    "bowler",
    "is_super_over",
    "batsman_runs",
    "extra_runs",
    "total_runs",
    "player_dismissed",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _validate_columns(df: pd.DataFrame, required: set[str], name: str) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"[{name}] Missing expected columns: {sorted(missing)}\n"
            f"Actual columns: {sorted(df.columns)}"
        )


def _abbreviate_teams(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = df[col].replace(TEAM_ABBREV)
    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_matches(path: str | Path | None = None) -> pd.DataFrame:
    """
    Load and clean matches.csv.

    Parameters
    ----------
    path : optional override; defaults to DATA_DIR/matches.csv
    """
    path = Path(path) if path else DATA_DIR / "matches.csv"
    logger.info("Loading matches from %s", path)

    df = pd.read_csv(path)
    _validate_columns(df, MATCHES_REQUIRED_COLS, "matches.csv")

    # Drop umpire3 if present (almost always empty)
    if "umpire3" in df.columns:
        df.drop(columns=["umpire3"], inplace=True)

    # Parse date
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Abbreviate team names in every relevant column
    team_cols = ["team1", "team2", "toss_winner", "winner"]
    df = _abbreviate_teams(df, team_cols)

    # city: fill NaN from venue.  Venue strings are typically
    # "Stadium Name, City" so we take the *last* comma-separated token.
    # For single-token venues (no comma) str[-1] returns the whole string,
    # which is no worse than the previous behaviour.
    if "city" in df.columns:
        df["city"] = df["city"].fillna(df["venue"].str.split(",").str[-1].str.strip())

    # Derived column: did the toss winner also win the match?
    df["toss_match_winner"] = df["toss_winner"] == df["winner"]

    logger.info("matches loaded: %d rows, %d cols", *df.shape)
    return df


def load_deliveries(path: str | Path | None = None) -> pd.DataFrame:
    """
    Load and clean deliveries.csv.

    Parameters
    ----------
    path : optional override; defaults to DATA_DIR/deliveries.csv
    """
    path = Path(path) if path else DATA_DIR / "deliveries.csv"
    logger.info("Loading deliveries from %s", path)

    df = pd.read_csv(path)
    _validate_columns(df, DELIVERIES_REQUIRED_COLS, "deliveries.csv")

    # player_dismissed NaN = batsman not out; keep as NaN, do NOT fill with 0
    # Only fill numeric extras with 0 where appropriate
    numeric_zero_fill = [
        "wide_runs",
        "bye_runs",
        "legbye_runs",
        "noball_runs",
        "penalty_runs",
        "extra_runs",
    ]
    for col in numeric_zero_fill:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(np.int8)

    # is_super_over: coerce to bool
    df["is_super_over"] = df["is_super_over"].astype(bool)

    # Validate over indexing ------------------------------------------------
    # All downstream feature engineering assumes 1-indexed overs (1–20).
    # The Kaggle nowke9/ipldata dataset uses 1-indexed overs; log a clear
    # warning immediately if the loaded file differs so the issue surfaces at
    # load time rather than silently corrupting phase bins and RR calculations.
    over_min = int(df["over"].min())
    over_max = int(df["over"].max())
    if over_min == 0:
        logger.warning(
            "deliveries 'over' column appears to be 0-indexed "
            "(min=%d, max=%d). features.py assumes 1-indexed overs (1-20). "
            "Phase bins, balls_completed, and score-prediction filters will "
            "be incorrect. Adjust bins and formulas before proceeding.",
            over_min,
            over_max,
        )
    elif over_min != 1:
        logger.warning(
            "Unexpected over minimum: %d (expected 1 for 1-indexed overs). "
            "Verify that feature engineering assumptions still hold.",
            over_min,
        )
    # -----------------------------------------------------------------------

    # Abbreviate team names
    team_cols = ["batting_team", "bowling_team"]
    df = _abbreviate_teams(df, team_cols)

    # Derived: is this a legal delivery (not wide/no-ball)?
    if "wide_runs" in df.columns and "noball_runs" in df.columns:
        df["is_legal_delivery"] = (df["wide_runs"] == 0) & (df["noball_runs"] == 0)

    # Derived: dismissal flag
    df["is_wicket"] = df["player_dismissed"].notna() & (df["player_dismissed"] != 0)

    logger.info("deliveries loaded: %d rows, %d cols", *df.shape)
    return df


def load_both(
    matches_path: str | Path | None = None,
    deliveries_path: str | Path | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Convenience wrapper: loads and returns (matches, deliveries)."""
    matches = load_matches(matches_path)
    deliveries = load_deliveries(deliveries_path)
    return matches, deliveries


def save_processed(
    df: pd.DataFrame,
    name: str,
    out_dir: str | Path | None = None,
) -> Path:
    """Save a cleaned DataFrame to data/processed/."""
    out_dir = (
        Path(out_dir)
        if out_dir
        else Path(__file__).resolve().parents[1] / "data" / "processed"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}.parquet"
    df.to_parquet(out_path, index=False)
    logger.info("Saved %s → %s", name, out_path)
    return out_path
