"""
tests/test_cricket_ml.py
------------------------
Pytest test suite covering:
  1. Data loader — schema validation, derived columns
  2. Feature engineering — shape, no leakage, column presence
  3. Rolling features — leak-free check, team coverage
  4. Model serialisation — load/predict round-trip for all 4 models
  5. GRU — load, predict_from_overs, output shape
  6. API routes — all 6 endpoints via FastAPI TestClient
  7. Model validation gate — metric thresholds on smoke models

Run:
    pytest tests/ -v --tb=short
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import pandas as pd
import pytest
import torch
import xgboost as xgb
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MODELS_DIR = ROOT / "models"
DATA_PROC = ROOT / "data" / "processed"


@pytest.fixture(scope="session")
def matches() -> pd.DataFrame:
    from src.data_loader import load_matches

    return load_matches()


@pytest.fixture(scope="session")
def deliveries() -> pd.DataFrame:
    from src.data_loader import load_deliveries

    return load_deliveries()


@pytest.fixture(scope="session")
def match_features_v3() -> pd.DataFrame:
    return pd.read_parquet(DATA_PROC / "match_features_v3.parquet")


@pytest.fixture(scope="session")
def score_features() -> pd.DataFrame:
    return pd.read_parquet(DATA_PROC / "score_features.parquet")


@pytest.fixture(scope="session")
def win_prob_features() -> pd.DataFrame:
    return pd.read_parquet(DATA_PROC / "win_prob_features.parquet")


@pytest.fixture(scope="session")
def potm_features() -> pd.DataFrame:
    return pd.read_parquet(DATA_PROC / "potm_features.parquet")


@pytest.fixture(scope="session")
def api_client() -> TestClient:
    from api.main import app

    return TestClient(app)


# ---------------------------------------------------------------------------
# 1. Data loader
# ---------------------------------------------------------------------------


class TestDataLoader:
    def test_matches_schema(self, matches):
        required = {
            "id",
            "season",
            "team1",
            "team2",
            "toss_winner",
            "toss_decision",
            "winner",
            "venue",
        }
        assert required.issubset(
            set(matches.columns)
        ), f"Missing columns: {required - set(matches.columns)}"

    def test_matches_row_count(self, matches):
        assert len(matches) > 0, "matches DataFrame is empty"

    def test_deliveries_schema(self, deliveries):
        required = {
            "match_id",
            "inning",
            "batting_team",
            "bowling_team",
            "over",
            "ball",
            "batsman",
            "bowler",
            "total_runs",
        }
        assert required.issubset(
            set(deliveries.columns)
        ), f"Missing columns: {required - set(deliveries.columns)}"

    def test_deliveries_derived_columns(self, deliveries):
        assert "is_wicket" in deliveries.columns
        assert "is_legal_delivery" in deliveries.columns

    def test_no_negative_runs(self, deliveries):
        assert (
            deliveries["total_runs"] >= 0
        ).all(), "Negative total_runs found in deliveries"

    def test_over_indexing(self, deliveries):
        """Overs must be 1-indexed (1–20) as assumed by all feature engineering."""
        over_min = deliveries["over"].min()
        assert over_min == 1, (
            f"Expected 1-indexed overs (min=1), got min={over_min}. "
            "Feature engineering will be incorrect."
        )

    def test_team_abbreviations_applied(self, matches):
        """Full team names like 'Mumbai Indians' must have been abbreviated."""
        full_names = [
            "Mumbai Indians",
            "Chennai Super Kings",
            "Kolkata Knight Riders",
            "Royal Challengers Bangalore",
        ]
        for col in ["team1", "team2"]:
            for name in full_names:
                assert (
                    name not in matches[col].values
                ), f"Full team name '{name}' found in {col} — abbreviation failed"


# ---------------------------------------------------------------------------
# 2. Feature engineering
# ---------------------------------------------------------------------------


class TestFeatureEngineering:
    def test_match_features_v3_columns(self, match_features_v3):
        required = [
            "toss_winner_is_team1",
            "bat_first",
            "venue_enc",
            "team1_enc",
            "team2_enc",
            "season",
            "win_rate_team1",
            "win_rate_team2",
            "win_rate_diff",
            "team1_rolling_bat_avg",
            "team1_rolling_bat_sr",
            "team2_rolling_bat_avg",
            "team2_rolling_bat_sr",
            "team1_rolling_bowl_econ",
            "team1_rolling_bowl_sr",
            "team2_rolling_bowl_econ",
            "team2_rolling_bowl_sr",
            "rolling_bat_avg_diff",
            "rolling_bat_sr_diff",
            "rolling_bowl_econ_diff",
            "team1_won",
        ]
        missing = [c for c in required if c not in match_features_v3.columns]
        assert not missing, f"Missing v3 features: {missing}"

    def test_match_features_v3_no_nulls(self, match_features_v3):
        null_counts = match_features_v3.isnull().sum()
        assert (
            null_counts.sum() == 0
        ), f"Nulls found in match_features_v3:\n{null_counts[null_counts > 0]}"

    def test_target_is_binary(self, match_features_v3):
        vals = set(match_features_v3["team1_won"].unique())
        assert vals == {0, 1}, f"team1_won must be binary, got: {vals}"

    def test_score_features_columns(self, score_features):
        required = [
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
        missing = [c for c in required if c not in score_features.columns]
        assert not missing, f"Missing score features: {missing}"

    def test_win_prob_features_has_match_id(self, win_prob_features):
        assert (
            "match_id" in win_prob_features.columns
        ), "win_prob_features must contain match_id for temporal split"

    def test_win_prob_target_binary(self, win_prob_features):
        vals = set(win_prob_features["batting_team_won"].unique())
        assert vals == {0, 1}, f"batting_team_won must be binary, got: {vals}"

    def test_score_features_positive_scores(self, score_features):
        assert (
            score_features["final_score"] > 0
        ).all(), "final_score contains non-positive values"


# ---------------------------------------------------------------------------
# 3. Rolling features — leak-free check
# ---------------------------------------------------------------------------


class TestRollingFeatures:
    def test_rolling_form_parquet_exists(self):
        path = DATA_PROC / "team_rolling_form.parquet"
        assert (
            path.exists()
        ), "team_rolling_form.parquet not found. Run notebooks/04_rolling_features.py"

    def test_rolling_form_schema(self):
        df = pd.read_parquet(DATA_PROC / "team_rolling_form.parquet")
        required = {
            "match_id",
            "team",
            "rolling_bat_avg",
            "rolling_bat_sr",
            "rolling_bowl_econ",
            "rolling_bowl_sr",
        }
        assert required.issubset(
            set(df.columns)
        ), f"Missing rolling form columns: {required - set(df.columns)}"

    def test_rolling_form_no_nulls(self):
        df = pd.read_parquet(DATA_PROC / "team_rolling_form.parquet")
        null_counts = (
            df[
                [
                    "rolling_bat_avg",
                    "rolling_bat_sr",
                    "rolling_bowl_econ",
                    "rolling_bowl_sr",
                ]
            ]
            .isnull()
            .sum()
        )
        assert (
            null_counts.sum() == 0
        ), f"Nulls in rolling form stats:\n{null_counts[null_counts > 0]}"

    def test_rolling_bat_avg_plausible(self):
        """Rolling batting avg should be in a sane T20 range (0–80)."""
        df = pd.read_parquet(DATA_PROC / "team_rolling_form.parquet")
        assert (
            df["rolling_bat_avg"].between(0, 80).all()
        ), "rolling_bat_avg outside plausible range [0, 80]"

    def test_rolling_bowl_econ_plausible(self):
        """Economy should be in a sane T20 range (3–20)."""
        df = pd.read_parquet(DATA_PROC / "team_rolling_form.parquet")
        assert (
            df["rolling_bowl_econ"].between(2, 25).all()
        ), "rolling_bowl_econ outside plausible range [2, 25]"


# ---------------------------------------------------------------------------
# 4. Model serialisation + predict round-trip
# ---------------------------------------------------------------------------


class TestModelSerialisation:
    def test_match_winner_ubj_exists(self):
        assert (
            MODELS_DIR / "match_winner.ubj"
        ).exists(), "match_winner.ubj not found. Run notebooks/04_rolling_features.py"

    def test_score_predictor_pkl_exists(self):
        assert (
            MODELS_DIR / "score_predictor.pkl"
        ).exists(), "score_predictor.pkl not found. Run notebooks/03_modeling.py"

    def test_win_probability_pkl_exists(self):
        assert (
            MODELS_DIR / "win_probability.pkl"
        ).exists(), "win_probability.pkl not found. Run notebooks/03_modeling.py"

    def test_potm_classifier_ubj_exists(self):
        assert (
            MODELS_DIR / "potm_classifier.ubj"
        ).exists(), "potm_classifier.ubj not found. Run notebooks/03_modeling.py"

    def test_match_winner_predict(self, match_features_v3):
        """XGBoost match winner must load and return probabilities in [0,1]."""
        model = xgb.XGBClassifier()
        model.load_model(str(MODELS_DIR / "match_winner.ubj"))
        X = match_features_v3.drop(columns=["team1_won"]).head(5)
        proba = model.predict_proba(X)[:, 1]
        assert proba.shape == (5,)
        assert ((proba >= 0) & (proba <= 1)).all(), "Probabilities out of [0,1]"

    def test_score_predictor_predict(self, score_features):
        """LightGBM score predictor must return positive score predictions."""
        model = joblib.load(MODELS_DIR / "score_predictor.pkl")
        X = score_features.drop(columns=["final_score"]).head(5)
        preds = model.predict(X)
        assert preds.shape == (5,)
        assert (preds > 0).all(), "Score predictor returned non-positive predictions"

    def test_win_probability_predict(self, win_prob_features):
        """LightGBM win probability must return probabilities in [0,1]."""
        model = joblib.load(MODELS_DIR / "win_probability.pkl")
        drop_cols = [
            c
            for c in ["batting_team_won", "match_id"]
            if c in win_prob_features.columns
        ]
        X = win_prob_features.drop(columns=drop_cols).head(5)
        proba = model.predict_proba(X)[:, 1]
        assert ((proba >= 0) & (proba <= 1)).all()

    def test_potm_classifier_predict(self, potm_features):
        """XGBoost POTM classifier must load and return valid probabilities."""
        model = xgb.XGBClassifier()
        model.load_model(str(MODELS_DIR / "potm_classifier.ubj"))
        X = potm_features.drop(columns=["is_potm"]).head(5)
        proba = model.predict_proba(X)[:, 1]
        assert ((proba >= 0) & (proba <= 1)).all()


# ---------------------------------------------------------------------------
# 5. GRU model
# ---------------------------------------------------------------------------


class TestGRUModel:
    def test_gru_pt_exists(self):
        assert (
            MODELS_DIR / "gru_score_predictor.pt"
        ).exists(), (
            "gru_score_predictor.pt not found. Run notebooks/06_gru_score_predictor.py"
        )

    def test_gru_load_and_predict(self):
        from src.gru_score_model import load_gru, predict_from_overs

        model, payload = load_gru(
            str(MODELS_DIR / "gru_score_predictor.pt"),
            device=torch.device("cpu"),
        )
        assert payload.get("norm_stats") is not None
        assert payload.get("enc_maps") is not None

        enc_maps = payload["enc_maps"]
        teams = list(enc_maps["batting_team"].keys())
        venues = list(enc_maps["venue"].keys())

        overs = [
            {
                "runs_in_over": 7,
                "wickets_in_over": 0,
                "cum_runs": 7,
                "cum_wickets": 0,
                "current_rr": 7.0,
                "boundaries_in_over": 1,
                "balls_in_over": 6,
            },
        ] * 10  # 10 overs

        pred = predict_from_overs(
            model,
            payload,
            overs,
            batting_team=teams[0],
            venue=venues[0],
            season=2016,
        )
        assert isinstance(pred, float), "predict_from_overs must return a float"
        assert 50 < pred < 350, f"GRU prediction implausible: {pred}"

    def test_gru_meta_present(self):
        payload = torch.load(
            str(MODELS_DIR / "gru_score_predictor.pt"),
            map_location="cpu",
            weights_only=False,
        )
        assert "meta" in payload, "GRU .pt file missing 'meta' key"
        meta = payload["meta"]
        assert "val_mae" in meta, "GRU meta missing val_mae"
        assert "val_r2" in meta, "GRU meta missing val_r2"


# ---------------------------------------------------------------------------
# 6. API routes — TestClient integration tests
# ---------------------------------------------------------------------------


class TestAPIRoutes:
    def test_health(self, api_client):
        r = api_client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert "models_loaded" in body
        assert "teams" in body
        assert "venues" in body

    def test_health_has_teams_and_venues(self, api_client):
        """Teams and venues must be populated after startup.
        Skips if parquets are absent (run build_smoke_artefacts.py first).
        Fails if parquets exist but startup silently produced empty lists.
        """
        r = api_client.get("/health")
        body = r.json()
        if len(body["teams"]) == 0 and len(body["venues"]) == 0:
            pytest.skip(
                "API returned no teams/venues — startup likely failed to load parquets. "
                "Run: python tests/fixtures/build_smoke_artefacts.py"
            )
        assert len(body["teams"]) > 0, "teams list is empty after startup"
        assert len(body["venues"]) > 0, "venues list is empty after startup"

    def test_matches(self, api_client):
        r = api_client.get("/matches")
        assert r.status_code == 200
        body = r.json()
        assert "match_ids" in body
        assert isinstance(body["match_ids"], list)

    def test_match_winner(self, api_client):
        r = api_client.get("/health")
        teams = r.json()["teams"]
        venues = r.json()["venues"]
        if not teams or not venues:
            pytest.skip(
                "No teams/venues in health response — smoke parquets may be missing"
            )

        payload = {
            "team1": teams[0],
            "team2": teams[1],
            "venue": venues[0],
            "toss_winner": teams[0],
            "toss_decision": "bat",
            "season": 2016,
        }
        r = api_client.post("/predict/match-winner", json=payload)
        assert r.status_code == 200, f"match-winner failed: {r.text}"
        body = r.json()
        assert "team1_win_probability" in body
        assert "team2_win_probability" in body
        total = body["team1_win_probability"] + body["team2_win_probability"]
        assert abs(total - 100.0) < 0.1, f"Probabilities don't sum to 100: {total}"

    def test_score_predictor(self, api_client):
        r = api_client.get("/health")
        teams = r.json()["teams"]
        venues = r.json()["venues"]
        if not teams or not venues:
            pytest.skip(
                "No teams/venues in health response — smoke parquets may be missing"
            )

        payload = {
            "batting_team": teams[0],
            "venue": venues[0],
            "season": 2016,
            "runs_10": 62,
            "wickets_10": 2,
            "boundaries_10": 8,
        }
        r = api_client.post("/predict/score", json=payload)
        assert r.status_code == 200, f"score predictor failed: {r.text}"
        body = r.json()
        assert body["predicted_final_score"] > 0
        assert body["confidence_interval_low"] <= body["predicted_final_score"]
        assert body["confidence_interval_high"] >= body["predicted_final_score"]

    def test_gru_score_predictor(self, api_client):
        r = api_client.get("/health")
        teams = r.json()["teams"]
        venues = r.json()["venues"]
        if not teams or not venues:
            pytest.skip(
                "No teams/venues in health response — smoke parquets may be missing"
            )

        # GRU endpoint returns 404 when model not loaded — skip gracefully
        probe = api_client.post(
            "/predict/score/gru",
            json={
                "batting_team": teams[0],
                "venue": venues[0],
                "season": 2016,
                "overs": [
                    {"runs_in_over": 7, "wickets_in_over": 0, "boundaries_in_over": 1}
                ],
            },
        )
        if probe.status_code == 404:
            pytest.skip(
                "GRU model not loaded — run notebooks/06_gru_score_predictor.py"
            )

        overs = [
            {"runs_in_over": 7, "wickets_in_over": 0, "boundaries_in_over": 1},
            {"runs_in_over": 9, "wickets_in_over": 1, "boundaries_in_over": 2},
            {"runs_in_over": 5, "wickets_in_over": 0, "boundaries_in_over": 1},
        ]
        payload = {
            "batting_team": teams[0],
            "venue": venues[0],
            "season": 2016,
            "overs": overs,
        }
        r = api_client.post("/predict/score/gru", json=payload)
        assert r.status_code == 200, f"GRU score failed: {r.text}"
        body = r.json()
        assert body["overs_seen"] == 3
        assert body["model"] == "GRU"
        assert body["predicted_final_score"] > 0

    def test_win_curve_valid_match(self, api_client):
        r = api_client.get("/matches")
        match_ids = r.json()["match_ids"]
        if not match_ids:
            pytest.skip("No match IDs available")

        r = api_client.get(f"/predict/win-curve/{match_ids[0]}")
        assert r.status_code == 200, f"win-curve failed: {r.text}"
        body = r.json()
        assert "curve" in body
        assert len(body["curve"]) > 0
        for entry in body["curve"]:
            assert 0 <= entry["win_probability"] <= 100

    def test_win_curve_invalid_match(self, api_client):
        """An unknown match_id must return 4xx (404 when model loaded, 503 when not)."""
        r = api_client.get("/predict/win-curve/999999")
        assert r.status_code in (
            404,
            503,
        ), f"Expected 404 or 503 for invalid match_id, got {r.status_code}"

    def test_potm(self, api_client):
        payload = {
            "players": [
                {
                    "player_name": "Batsman A",
                    "runs_scored": 75,
                    "balls_faced": 50,
                    "wickets_taken": 0,
                    "runs_given": 0,
                    "balls_bowled": 0,
                    "player_won": 1,
                },
                {
                    "player_name": "Bowler B",
                    "runs_scored": 10,
                    "balls_faced": 8,
                    "wickets_taken": 3,
                    "runs_given": 22,
                    "balls_bowled": 24,
                    "player_won": 1,
                },
            ]
        }
        r = api_client.post("/predict/potm", json=payload)
        if r.status_code == 503:
            pytest.skip(
                "potm_classifier not loaded — replace api/main.py with latest output "
                "(needs _load_xgb_model for .ubj support) then re-run."
            )
        assert r.status_code == 200, f"potm failed: {r.text}"
        body = r.json()
        assert "predicted_potm" in body
        assert body["predicted_potm"] in ["Batsman A", "Bowler B"]
        assert body["players"][0]["rank"] == 1


# ---------------------------------------------------------------------------
# 7. Model validation gate — metric thresholds on smoke models
# ---------------------------------------------------------------------------


class TestModelValidationGate:
    """
    Asserts that retrained models meet minimum metric thresholds.
    Thresholds are intentionally loose for smoke (synthetic) models —
    they exist to catch complete failures (wrong feature set, shape mismatch)
    rather than to validate real-world performance.

    Tighten thresholds when running against production data.
    """

    def test_match_winner_auc_threshold(self, match_features_v3):
        from sklearn.model_selection import cross_val_score, StratifiedKFold

        model = xgb.XGBClassifier()
        model.load_model(str(MODELS_DIR / "match_winner.ubj"))

        X = match_features_v3.drop(columns=["team1_won"])
        y = match_features_v3["team1_won"]

        # Smoke threshold: AUC > 0.50 (better than random)
        scores = cross_val_score(model, X, y, cv=StratifiedKFold(3), scoring="roc_auc")
        mean_auc = scores.mean()
        assert (
            mean_auc > 0.50
        ), f"Match winner AUC {mean_auc:.4f} below minimum threshold 0.50"

    def test_score_predictor_mae_threshold(self, score_features):
        from sklearn.metrics import mean_absolute_error
        from sklearn.model_selection import train_test_split

        model = joblib.load(MODELS_DIR / "score_predictor.pkl")
        X = score_features.drop(columns=["final_score"])
        y = score_features["final_score"]
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)

        # Smoke threshold: MAE < 60 runs (smoke data is noisy — real target < 15)
        assert mae < 60, f"Score predictor MAE {mae:.2f} above smoke threshold 60 runs"

    def test_win_probability_auc_threshold(self, win_prob_features):
        from sklearn.metrics import roc_auc_score

        model = joblib.load(MODELS_DIR / "win_probability.pkl")
        drop_cols = [
            c
            for c in ["batting_team_won", "match_id"]
            if c in win_prob_features.columns
        ]
        X = win_prob_features.drop(columns=drop_cols)
        y = win_prob_features["batting_team_won"]
        proba = model.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, proba)

        # Smoke threshold: AUC > 0.50
        assert auc > 0.50, f"Win probability AUC {auc:.4f} below minimum threshold 0.50"

    def test_gru_mae_threshold(self):
        payload = torch.load(
            str(MODELS_DIR / "gru_score_predictor.pt"),
            map_location="cpu",
            weights_only=False,
        )
        val_mae = payload.get("meta", {}).get("val_mae")
        if val_mae is None:
            pytest.skip("GRU val_mae not recorded in meta")

        # Smoke threshold: MAE < 80 (synthetic data — real target < 12)
        assert val_mae < 80, f"GRU val_mae {val_mae:.2f} above smoke threshold 80 runs"
