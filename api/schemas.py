"""
api/schemas.py
--------------
Pydantic request / response contracts for the Cricket ML API.

Encoding note
-------------
All *_enc features in the trained models were produced by
    pd.Series.astype("category").cat.codes
which assigns codes in alphabetically-sorted order of unique values seen in
the training set.  The API reconstructs these mappings at startup from the
processed parquets and stores them in a CricketContext object; the schemas
here deal only with human-readable inputs.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator

# ---------------------------------------------------------------------------
# 1. Match Winner
# ---------------------------------------------------------------------------


class MatchWinnerRequest(BaseModel):
    team1: str = Field(..., examples=["MI"])
    team2: str = Field(..., examples=["CSK"])
    venue: str = Field(..., examples=["Wankhede Stadium"])
    toss_winner: Literal["team1", "team2"] = Field(
        ..., description="Which team won the toss"
    )
    toss_decision: Literal["bat", "field"]
    season: int = Field(..., ge=2008, le=2030, examples=[2023])


class MatchWinnerResponse(BaseModel):
    team1: str
    team2: str
    team1_win_probability: float = Field(..., description="0–100 probability for team1")
    team2_win_probability: float
    model_version: str = "match_winner_v1"


# ---------------------------------------------------------------------------
# 2. Score Predictor
# ---------------------------------------------------------------------------


class ScorePredictorRequest(BaseModel):
    """State at the end of over 10 in the 1st innings."""

    batting_team: str = Field(..., examples=["MI"])
    venue: str = Field(..., examples=["Wankhede Stadium"])
    season: int = Field(..., ge=2008, le=2030)
    runs_10: int = Field(..., ge=0, le=150, description="Runs scored in first 10 overs")
    wickets_10: int = Field(..., ge=0, le=10)
    boundaries_10: int = Field(
        ..., ge=0, description="Count of 4s and 6s in first 10 overs"
    )


class ScorePredictorResponse(BaseModel):
    predicted_final_score: float
    confidence_interval_low: float
    confidence_interval_high: float
    current_rr: float
    projected_naive: float = Field(..., description="Naive projection: current_rr × 20")
    model_version: str = "score_predictor_v1"


# ---------------------------------------------------------------------------
# 3. Win Probability Curve
# ---------------------------------------------------------------------------


class WinProbOverEntry(BaseModel):
    over: int
    win_probability: float  # 0–100, batting team's probability


class WinCurveResponse(BaseModel):
    match_id: int
    batting_team: str
    bowling_team: str
    curve: List[WinProbOverEntry]
    actual_winner: Optional[str] = None
    model_version: str = "win_probability_v1"


class AvailableMatchesResponse(BaseModel):
    match_ids: List[int]
    count: int


# ---------------------------------------------------------------------------
# 4. POTM Classifier
# ---------------------------------------------------------------------------


class PotmPlayerInput(BaseModel):
    player_name: str
    runs_scored: int = Field(..., ge=0)
    balls_faced: int = Field(..., ge=0)
    wickets_taken: int = Field(..., ge=0)
    runs_given: int = Field(..., ge=0, description="Runs conceded while bowling")
    balls_bowled: int = Field(..., ge=0)
    player_won: Literal[0, 1] = Field(
        ..., description="1 if player's team won the match"
    )

    @model_validator(mode="after")
    def _compute_rates(self) -> "PotmPlayerInput":
        # Derived fields are computed server-side; just validate inputs here.
        if self.balls_faced == 0 and self.runs_scored > 0:
            raise ValueError("runs_scored > 0 but balls_faced == 0")
        return self


class PotmRequest(BaseModel):
    players: List[PotmPlayerInput] = Field(..., min_length=1, max_length=22)


class PotmPlayerResult(BaseModel):
    player_name: str
    potm_probability: float  # 0–100
    strike_rate: float
    economy: float
    rank: int


class PotmResponse(BaseModel):
    predicted_potm: str
    players: List[PotmPlayerResult]
    model_version: str = "potm_classifier_v1"


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    status: str
    models_loaded: Dict[str, bool]
    teams: List[str]
    venues: List[str]


# ---------------------------------------------------------------------------
# GRU score predictor
# ---------------------------------------------------------------------------


class OverInput(BaseModel):
    """Stats for one completed over — values are for that over only (not cumulative)."""

    runs_in_over: int = Field(..., ge=0, le=36)
    wickets_in_over: int = Field(..., ge=0, le=10)
    boundaries_in_over: int = Field(..., ge=0)


class GruScoreRequest(BaseModel):
    """
    Predict final 1st-innings score from any number of completed overs (1–20).
    Cumulative features (cum_runs, current_rr, etc.) are derived server-side.
    """

    batting_team: str = Field(..., description="Batting team (abbreviated, e.g. 'MI')")
    venue: str
    season: int = Field(..., ge=2008, le=2030)
    overs: List[OverInput] = Field(..., min_length=1, max_length=20)


class GruScoreResponse(BaseModel):
    predicted_final_score: float
    confidence_interval_low: float
    confidence_interval_high: float
    overs_seen: int
    model: str = "GRU"
