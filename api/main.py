"""
api/main.py
-----------
FastAPI backend for the Cricket ML project.

Startup sequence
----------------
1. Load processed parquets from data/processed/ (matches, deliveries,
   win_prob_features).
2. Derive category encoding maps from the same sorted-unique values that
   pd.astype("category").cat.codes uses — so inference encoding matches
   training encoding exactly.
3. Precompute venue_avg_rr for the score predictor's scoring_pressure feature.
4. Load all four .pkl models.
5. Expose 6 HTTP routes.

Run with:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# ---------------------------------------------------------------------------
# Paths — resolve relative to this file so the app works from any cwd
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("cricket_api")

# ---------------------------------------------------------------------------
# Context — all shared state loaded once at startup
# ---------------------------------------------------------------------------


@dataclass
class CricketContext:
    # Raw data
    matches: pd.DataFrame = field(default_factory=pd.DataFrame)
    deliveries: pd.DataFrame = field(default_factory=pd.DataFrame)
    win_prob_features: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Models
    match_winner: Optional[Any] = None
    score_predictor: Optional[Any] = None
    win_probability: Optional[Any] = None
    potm_classifier: Optional[Any] = None
    gru_score_predictor: Optional[Any] = None  # GRU model (None until notebook 06 run)
    gru_payload: Optional[Dict] = None  # norm_stats, enc_maps, season_range

    # Encoding maps: human label → integer code
    # Built from sorted(unique values) to reproduce cat.codes behaviour
    venue_enc_map: Dict[str, int] = field(default_factory=dict)
    team_enc_map: Dict[str, int] = field(default_factory=dict)  # team1/team2 in matches
    batting_team_enc_map: Dict[str, int] = field(
        default_factory=dict
    )  # deliveries teams

    # Precomputed venue avg RR for scoring_pressure feature
    venue_avg_rr: Dict[str, float] = field(default_factory=dict)
    global_avg_rr: float = 7.5

    # Lookup tables
    match_id_list: List[int] = field(default_factory=list)
    teams: List[str] = field(default_factory=list)
    venues: List[str] = field(default_factory=list)

    # Rolling team form — populated after running notebooks/04_rolling_features.py.
    # Key: abbreviated team name → {rolling_bat_avg, rolling_bat_sr,
    #                                rolling_bowl_econ, rolling_bowl_sr}
    # Empty dict = API falls back to v2 features automatically.
    team_last_form: Dict[str, Dict[str, float]] = field(default_factory=dict)


ctx = CricketContext()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_enc_map(series: pd.Series) -> Dict[str, int]:
    """
    Reproduce pd.Series.astype("category").cat.codes.
    cat.codes assigns codes in the order categories are stored, which after
    astype("category") on a string column is alphabetical order.
    """
    unique_sorted = sorted(series.dropna().unique().tolist())
    return {label: code for code, label in enumerate(unique_sorted)}


def _encode(label: str, enc_map: Dict[str, int], feature_name: str) -> int:
    if label not in enc_map:
        raise HTTPException(
            status_code=422,
            detail=(
                f"'{label}' is not a recognised {feature_name}. "
                f"Valid values: {sorted(enc_map.keys())}"
            ),
        )
    return enc_map[label]


def _load_parquet(name: str) -> pd.DataFrame:
    path = PROCESSED_DIR / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Processed file not found: {path}. "
            "Run notebooks/02_feature_engineering.py first."
        )
    return pd.read_parquet(path)


def _load_model(name: str) -> Any:
    path = MODELS_DIR / f"{name}.pkl"
    if not path.exists():
        logger.warning("Model not found: %s", path)
        return None
    model = joblib.load(path)
    logger.info("Loaded model: %s (pkl)", name)
    return model


def _load_xgb_model(name: str) -> Any:
    """
    Load an XGBoost model saved in native .ubj format.
    Falls back to .pkl if .ubj is not present (backwards-compatible during
    the transition before retraining).
    """
    import xgboost as xgb

    ubj_path = MODELS_DIR / f"{name}.ubj"
    pkl_path = MODELS_DIR / f"{name}.pkl"

    if ubj_path.exists():
        model = xgb.XGBClassifier()
        model.load_model(str(ubj_path))
        logger.info("Loaded model: %s (ubj — no pickle warning)", name)
        return model

    if pkl_path.exists():
        logger.warning(
            "Loading %s from .pkl — run notebooks/03_modeling.py or "
            "notebooks/04_rolling_features.py to regenerate as .ubj "
            "and eliminate the XGBoost version warning.",
            name,
        )
        model = joblib.load(pkl_path)
        logger.info("Loaded model: %s (pkl fallback)", name)
        return model

    logger.warning("Model not found: %s (.ubj or .pkl)", name)
    return None


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------


def _startup() -> None:
    logger.info("=== Cricket API startup ===")

    # ── 1. Load parquets ──────────────────────────────────────────────────
    ctx.matches = _load_parquet("matches")
    ctx.deliveries = _load_parquet("deliveries")
    ctx.win_prob_features = _load_parquet("win_prob_features")
    logger.info(
        "Parquets loaded: matches=%d, deliveries=%d, win_prob=%d",
        len(ctx.matches),
        len(ctx.deliveries),
        len(ctx.win_prob_features),
    )

    # ── 2. Encoding maps ─────────────────────────────────────────────────
    # venue_enc: built from ALL venues seen in matches (same as training)
    ctx.venue_enc_map = _build_enc_map(ctx.matches["venue"])
    # team_enc for match winner (team1/team2 columns in matches)
    all_teams_in_matches = pd.concat([ctx.matches["team1"], ctx.matches["team2"]])
    ctx.team_enc_map = _build_enc_map(all_teams_in_matches)
    # batting_team_enc for score predictor and win probability
    ctx.batting_team_enc_map = _build_enc_map(ctx.deliveries["batting_team"])

    ctx.teams = sorted(ctx.team_enc_map.keys())
    ctx.venues = sorted(ctx.venue_enc_map.keys())
    logger.info(
        "Encoding maps: %d teams, %d venues",
        len(ctx.teams),
        len(ctx.venues),
    )

    # ── 3. venue_avg_rr ──────────────────────────────────────────────────
    # Recompute from match-level first-10-overs data (same logic as
    # build_score_features, but we only need the final average per venue,
    # not the per-row rolling value).
    is_first = (ctx.deliveries["inning"] == 1) & (
        ~ctx.deliveries["is_super_over"].astype(bool)
    )
    half = (
        ctx.deliveries[is_first & (ctx.deliveries["over"] <= 10)]
        .groupby("match_id")["total_runs"]
        .sum()
        .reset_index(name="runs_10")
    )
    half["current_rr"] = half["runs_10"] / 10

    match_venue = ctx.matches[["id", "venue"]].rename(columns={"id": "match_id"})
    half = half.merge(match_venue, on="match_id", how="left")
    venue_rr = half.groupby("venue")["current_rr"].mean().round(3)
    ctx.venue_avg_rr = venue_rr.to_dict()
    ctx.global_avg_rr = float(half["current_rr"].mean())
    logger.info("venue_avg_rr computed for %d venues", len(ctx.venue_avg_rr))

    # ── 4. Load models ────────────────────────────────────────────────────
    # XGBoost models: use native .ubj loader (no pickle version warning)
    # LightGBM models: joblib pickle is stable across versions
    ctx.match_winner = _load_xgb_model("match_winner")
    ctx.score_predictor = _load_model("score_predictor")
    ctx.win_probability = _load_model("win_probability")
    ctx.potm_classifier = _load_xgb_model("potm_classifier")

    # ── GRU score predictor (optional — requires notebook 06) ─────────────
    _gru_path = MODELS_DIR / "gru_score_predictor.pt"
    if _gru_path.exists():
        try:
            from src.gru_score_model import load_gru
            import torch as _torch

            _device = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
            ctx.gru_score_predictor, ctx.gru_payload = load_gru(
                str(_gru_path), device=_device
            )
            _meta = ctx.gru_payload.get("meta", {})
            logger.info(
                "Loaded GRU score predictor — MAE: %.2f  R²: %.4f",
                _meta.get("val_mae", float("nan")),
                _meta.get("val_r2", float("nan")),
            )
        except Exception as _e:
            logger.warning("Failed to load GRU score predictor: %s", _e)
    else:
        logger.warning(
            "gru_score_predictor.pt not found — /predict/score/gru will return 404. "
            "Run notebooks/06_gru_score_predictor.py to enable."
        )

    # ── 5. Match ID list for /matches endpoint ────────────────────────────
    if "match_id" in ctx.win_prob_features.columns:
        ctx.match_id_list = sorted(ctx.win_prob_features["match_id"].unique().tolist())
    logger.info(
        "=== Startup complete: %d matches available ===", len(ctx.match_id_list)
    )

    # ── 6. Rolling team form (optional — requires notebook 04) ────────────
    try:
        team_form_df = _load_parquet("team_rolling_form")
        # For each team, take stats from their most recent match
        latest_idx = team_form_df.groupby("team")["match_id"].idxmax()
        latest = team_form_df.loc[latest_idx]
        ctx.team_last_form = {
            str(row["team"]): {
                "rolling_bat_avg": float(row["rolling_bat_avg"]),
                "rolling_bat_sr": float(row["rolling_bat_sr"]),
                "rolling_bowl_econ": float(row["rolling_bowl_econ"]),
                "rolling_bowl_sr": float(row["rolling_bowl_sr"]),
            }
            for _, row in latest.iterrows()
        }
        logger.info(
            "Rolling team form loaded for %d teams (v3 match winner active)",
            len(ctx.team_last_form),
        )
    except FileNotFoundError:
        logger.warning(
            "team_rolling_form.parquet not found — match winner will use v2 features. "
            "Run notebooks/04_rolling_features.py to enable v3 (rolling player form)."
        )


# ---------------------------------------------------------------------------
# App — lifespan replaces deprecated @app.on_event("startup")
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run startup logic before yielding; re-raise so TestClient sees failures."""
    try:
        _startup()
    except Exception as exc:
        logger.error("Startup failed: %s", exc)
        raise  # propagate — prevents silent empty-ctx in tests
    yield


app = FastAPI(
    title="Cricket ML API",
    version="1.0.0",
    description="Inference API for four IPL ML models.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],  # Streamlit origin
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Import schemas after app is defined (avoids circular imports if schemas
# later import from this module)
# ---------------------------------------------------------------------------

from api.schemas import (  # noqa: E402
    AvailableMatchesResponse,
    GruScoreRequest,
    GruScoreResponse,
    HealthResponse,
    MatchWinnerRequest,
    MatchWinnerResponse,
    PotmRequest,
    PotmResponse,
    PotmPlayerResult,
    ScorePredictorRequest,
    ScorePredictorResponse,
    WinCurveResponse,
    WinProbOverEntry,
)

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse, tags=["Utility"])
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        models_loaded={
            "match_winner": ctx.match_winner is not None,
            "score_predictor": ctx.score_predictor is not None,
            "win_probability": ctx.win_probability is not None,
            "potm_classifier": ctx.potm_classifier is not None,
        },
        teams=ctx.teams,
        venues=ctx.venues,
    )


@app.get("/matches", response_model=AvailableMatchesResponse, tags=["Utility"])
def available_matches() -> AvailableMatchesResponse:
    return AvailableMatchesResponse(
        match_ids=ctx.match_id_list, count=len(ctx.match_id_list)
    )


# ── 1. Match Winner ───────────────────────────────────────────────────────


@app.post(
    "/predict/match-winner", response_model=MatchWinnerResponse, tags=["Prediction"]
)
def predict_match_winner(req: MatchWinnerRequest) -> MatchWinnerResponse:
    if ctx.match_winner is None:
        raise HTTPException(503, "match_winner model not loaded")

    # Resolve toss_winner to actual team name
    toss_winner_team = req.team1 if req.toss_winner == "team1" else req.team2
    toss_winner_is_team1 = int(toss_winner_team == req.team1)
    bat_first = int(req.toss_decision == "bat")

    venue_enc = _encode(req.venue, ctx.venue_enc_map, "venue")
    team1_enc = _encode(req.team1, ctx.team_enc_map, "team")
    team2_enc = _encode(req.team2, ctx.team_enc_map, "team")

    # Rolling win rates: compute from historical matches up to chosen season
    historical = ctx.matches[ctx.matches["season"] < req.season]

    def _win_rate(team: str) -> float:
        played = len(
            historical[(historical["team1"] == team) | (historical["team2"] == team)]
        )
        if played == 0:
            return 0.5
        won = len(historical[historical["winner"] == team])
        return round(won / played, 4)

    wr1 = _win_rate(req.team1)
    wr2 = _win_rate(req.team2)

    # ── Build feature vector ───────────────────────────────────────────────
    # V2 base features (always present)
    features: dict = {
        "toss_winner_is_team1": toss_winner_is_team1,
        "bat_first": bat_first,
        "venue_enc": venue_enc,
        "team1_enc": team1_enc,
        "team2_enc": team2_enc,
        "season": req.season,
        "win_rate_team1": wr1,
        "win_rate_team2": wr2,
        "win_rate_diff": round(wr1 - wr2, 4),
    }

    # V3 rolling features — injected only when notebook 04 has been run.
    # Falls back to v2 silently if team_rolling_form.parquet does not exist.
    if ctx.team_last_form:
        # IPL priors: roughly average for a T20 team across a season
        BAT_AVG_PRIOR = 25.0
        BAT_SR_PRIOR = 120.0
        BOWL_ECON_PRIOR = 8.0
        BOWL_SR_PRIOR = 20.0

        def _form(team: str, key: str, prior: float) -> float:
            return ctx.team_last_form.get(team, {}).get(key, prior)

        t1_bat_avg = _form(req.team1, "rolling_bat_avg", BAT_AVG_PRIOR)
        t1_bat_sr = _form(req.team1, "rolling_bat_sr", BAT_SR_PRIOR)
        t1_bowl_econ = _form(req.team1, "rolling_bowl_econ", BOWL_ECON_PRIOR)
        t1_bowl_sr = _form(req.team1, "rolling_bowl_sr", BOWL_SR_PRIOR)

        t2_bat_avg = _form(req.team2, "rolling_bat_avg", BAT_AVG_PRIOR)
        t2_bat_sr = _form(req.team2, "rolling_bat_sr", BAT_SR_PRIOR)
        t2_bowl_econ = _form(req.team2, "rolling_bowl_econ", BOWL_ECON_PRIOR)
        t2_bowl_sr = _form(req.team2, "rolling_bowl_sr", BOWL_SR_PRIOR)

        features.update(
            {
                "team1_rolling_bat_avg": round(t1_bat_avg, 3),
                "team1_rolling_bat_sr": round(t1_bat_sr, 3),
                "team2_rolling_bat_avg": round(t2_bat_avg, 3),
                "team2_rolling_bat_sr": round(t2_bat_sr, 3),
                "team1_rolling_bowl_econ": round(t1_bowl_econ, 3),
                "team1_rolling_bowl_sr": round(t1_bowl_sr, 3),
                "team2_rolling_bowl_econ": round(t2_bowl_econ, 3),
                "team2_rolling_bowl_sr": round(t2_bowl_sr, 3),
                "rolling_bat_avg_diff": round(t1_bat_avg - t2_bat_avg, 3),
                "rolling_bat_sr_diff": round(t1_bat_sr - t2_bat_sr, 3),
                "rolling_bowl_econ_diff": round(t1_bowl_econ - t2_bowl_econ, 3),
            }
        )

    X = pd.DataFrame([features])

    proba_team1 = float(ctx.match_winner.predict_proba(X)[0, 1])
    return MatchWinnerResponse(
        team1=req.team1,
        team2=req.team2,
        team1_win_probability=round(proba_team1 * 100, 1),
        team2_win_probability=round((1 - proba_team1) * 100, 1),
    )


# ── 2. Score Predictor ────────────────────────────────────────────────────


@app.post("/predict/score", response_model=ScorePredictorResponse, tags=["Prediction"])
def predict_score(req: ScorePredictorRequest) -> ScorePredictorResponse:
    if ctx.score_predictor is None:
        raise HTTPException(503, "score_predictor model not loaded")

    batting_team_enc = _encode(
        req.batting_team, ctx.batting_team_enc_map, "batting_team"
    )
    venue_enc = _encode(req.venue, ctx.venue_enc_map, "venue")

    current_rr = round(req.runs_10 / 10, 3)
    projected_score = round(current_rr * 20, 1)
    venue_avg_rr = ctx.venue_avg_rr.get(req.venue, ctx.global_avg_rr)
    scoring_pressure = round(current_rr - venue_avg_rr, 3)

    X = pd.DataFrame(
        [
            {
                "runs_10": req.runs_10,
                "wickets_10": req.wickets_10,
                "current_rr": current_rr,
                "projected_score": projected_score,
                "scoring_pressure": scoring_pressure,
                "boundaries_10": req.boundaries_10,
                "batting_team_enc": batting_team_enc,
                "venue_enc": venue_enc,
                "season": req.season,
            }
        ]
    )

    predicted = float(ctx.score_predictor.predict(X)[0])

    # Approximate 80% prediction interval using the model's training MAE.
    # MAE ≈ 13 runs (from training metrics); 1.28σ ≈ MAE for normal dist.
    # This is a heuristic; replace with conformal intervals if needed.
    MAE_HEURISTIC = 13.0
    return ScorePredictorResponse(
        predicted_final_score=round(predicted, 1),
        confidence_interval_low=round(predicted - MAE_HEURISTIC, 1),
        confidence_interval_high=round(predicted + MAE_HEURISTIC, 1),
        current_rr=current_rr,
        projected_naive=projected_score,
    )


# ── 3. Win Probability Curve ──────────────────────────────────────────────


@app.get(
    "/predict/win-curve/{match_id}",
    response_model=WinCurveResponse,
    tags=["Prediction"],
)
def predict_win_curve(match_id: int) -> WinCurveResponse:
    if ctx.win_probability is None:
        raise HTTPException(503, "win_probability model not loaded")

    if match_id not in set(ctx.match_id_list):
        raise HTTPException(
            404,
            f"match_id {match_id} not in win_prob_features. "
            f"Use GET /matches for valid IDs.",
        )

    match_data = ctx.win_prob_features[
        ctx.win_prob_features["match_id"] == match_id
    ].copy()

    drop_cols = ["batting_team_won", "match_id"]
    X = match_data.drop(columns=[c for c in drop_cols if c in match_data.columns])
    proba = ctx.win_probability.predict_proba(X)[:, 1]
    overs = match_data["over"].values.tolist()

    curve = [
        WinProbOverEntry(over=int(o), win_probability=round(float(p) * 100, 1))
        for o, p in zip(overs, proba)
    ]

    # Derive teams from deliveries
    match_deliveries = ctx.deliveries[
        (ctx.deliveries["match_id"] == match_id) & (ctx.deliveries["inning"] == 2)
    ]
    batting_team = (
        match_deliveries["batting_team"].iloc[0]
        if not match_deliveries.empty
        else "Unknown"
    )
    bowling_team = (
        match_deliveries["bowling_team"].iloc[0]
        if not match_deliveries.empty
        else "Unknown"
    )

    # Actual winner from matches
    match_row = ctx.matches[ctx.matches["id"] == match_id]
    actual_winner = str(match_row["winner"].iloc[0]) if not match_row.empty else None

    return WinCurveResponse(
        match_id=match_id,
        batting_team=batting_team,
        bowling_team=bowling_team,
        curve=curve,
        actual_winner=actual_winner,
    )


# ── 4. POTM Classifier ────────────────────────────────────────────────────


@app.post("/predict/potm", response_model=PotmResponse, tags=["Prediction"])
def predict_potm(req: PotmRequest) -> PotmResponse:
    if ctx.potm_classifier is None:
        raise HTTPException(503, "potm_classifier model not loaded")

    rows = []
    for p in req.players:
        sr = round(p.runs_scored / p.balls_faced * 100, 2) if p.balls_faced > 0 else 0.0
        eco = (
            round(p.runs_given / (p.balls_bowled / 6), 2) if p.balls_bowled > 0 else 0.0
        )
        rows.append(
            {
                "runs_scored": p.runs_scored,
                "balls_faced": p.balls_faced,
                "strike_rate": sr,
                "wickets_taken": p.wickets_taken,
                "runs_given": p.runs_given,
                "economy": eco,
                "player_won": p.player_won,
            }
        )

    X = pd.DataFrame(rows)
    # Feature order must match training
    feature_order = [
        "runs_scored",
        "balls_faced",
        "strike_rate",
        "wickets_taken",
        "runs_given",
        "economy",
        "player_won",
    ]
    X = X[feature_order]
    proba = ctx.potm_classifier.predict_proba(X)[:, 1]

    results = []
    for i, (p, prob) in enumerate(zip(req.players, proba)):
        sr = round(p.runs_scored / p.balls_faced * 100, 2) if p.balls_faced > 0 else 0.0
        eco = (
            round(p.runs_given / (p.balls_bowled / 6), 2) if p.balls_bowled > 0 else 0.0
        )
        results.append(
            PotmPlayerResult(
                player_name=p.player_name,
                potm_probability=round(float(prob) * 100, 1),
                strike_rate=sr,
                economy=eco,
                rank=0,  # filled below
            )
        )

    results.sort(key=lambda r: r.potm_probability, reverse=True)
    for rank, r in enumerate(results, start=1):
        r.rank = rank

    return PotmResponse(
        predicted_potm=results[0].player_name,
        players=results,
    )


# ---------------------------------------------------------------------------
# GRU score predictor endpoint
# ---------------------------------------------------------------------------


@app.post("/predict/score/gru", response_model=GruScoreResponse, tags=["Prediction"])
def predict_score_gru(req: GruScoreRequest) -> GruScoreResponse:
    """
    Predict first-innings final score using the GRU sequence model.

    Accepts per-over stats for all completed overs (1–20).
    Cumulative features are derived server-side.
    Returns a predicted score with a ±MAE confidence interval.

    Requires: run notebooks/06_gru_score_predictor.py first.
    """
    if ctx.gru_score_predictor is None or ctx.gru_payload is None:
        from fastapi import HTTPException

        raise HTTPException(
            status_code=404,
            detail=(
                "GRU score predictor not loaded. "
                "Run notebooks/06_gru_score_predictor.py then restart uvicorn."
            ),
        )

    from src.gru_score_model import predict_from_overs

    # Build over_rows with cumulative features derived server-side
    over_rows = []
    cum_runs = 0
    cum_wickets = 0
    for k, ov in enumerate(req.overs, start=1):
        cum_runs += ov.runs_in_over
        cum_wickets += ov.wickets_in_over
        over_rows.append(
            {
                "runs_in_over": ov.runs_in_over,
                "wickets_in_over": ov.wickets_in_over,
                "cum_runs": cum_runs,
                "cum_wickets": cum_wickets,
                "current_rr": round(cum_runs / k, 3),
                "boundaries_in_over": ov.boundaries_in_over,
                "balls_in_over": 6,  # API assumes full overs
            }
        )

    device = next(ctx.gru_score_predictor.parameters()).device
    predicted = predict_from_overs(
        ctx.gru_score_predictor,
        ctx.gru_payload,
        over_rows,
        batting_team=req.batting_team,
        venue=req.venue,
        season=req.season,
        device=device,
    )

    # Confidence interval: ±model MAE (calibrated on validation set)
    val_mae = ctx.gru_payload.get("meta", {}).get("val_mae", 13.0)
    predicted = round(predicted, 1)

    return GruScoreResponse(
        predicted_final_score=predicted,
        confidence_interval_low=round(max(predicted - val_mae, 0), 1),
        confidence_interval_high=round(predicted + val_mae, 1),
        overs_seen=len(req.overs),
    )
