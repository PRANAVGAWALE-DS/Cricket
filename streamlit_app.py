"""
streamlit_app.py
----------------
4-page Streamlit dashboard for the Cricket ML API.

Pages
-----
1. 🏏  Match Winner       — pre-match win probability gauge
2. 📊  Score Predictor    — over-10 snapshot → predicted final score
3. 📈  Win Probability    — over-by-over animated win curve for a historical match
4. 🏆  POTM Predictor     — player performance → Player of the Match probability

Run with:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")
REQUEST_TIMEOUT = 30  # seconds — first call after cold start can take 15–25 s

st.set_page_config(
    page_title="Cricket ML Dashboard",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Shared styles
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    .metric-card {
        background: #1e2130;
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 0.8rem;
    }
    .metric-card h2 { margin: 0; font-size: 2rem; }
    .metric-card p  { margin: 0; color: #aaa; font-size: 0.9rem; }
    .win-bar-container { height: 28px; border-radius: 6px; overflow: hidden;
                         display: flex; margin-top: 0.5rem; }
    .win-bar-t1 { background: #00b4d8; display: flex; align-items: center;
                  justify-content: center; color: white; font-weight: 700;
                  font-size: 0.85rem; }
    .win-bar-t2 { background: #ff6b6b; display: flex; align-items: center;
                  justify-content: center; color: white; font-weight: 700;
                  font-size: 0.85rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------


def _get(endpoint: str, **kwargs) -> Optional[Dict]:
    try:
        r = requests.get(f"{API_BASE}{endpoint}", timeout=REQUEST_TIMEOUT, **kwargs)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error(f"Cannot reach API at {API_BASE}. Is uvicorn running?")
    except requests.exceptions.Timeout:
        st.error("API request timed out. The server may be overloaded — try again.")
    except requests.exceptions.HTTPError as e:
        try:
            detail = e.response.json().get("detail", str(e))
        except Exception:
            detail = f"HTTP {e.response.status_code} — server returned a non-JSON error body. Check uvicorn logs."
        st.error(f"API error {e.response.status_code}: {detail}")
    except Exception as e:
        st.error(f"Unexpected error: {e}")
    return None


def _post(endpoint: str, payload: Dict) -> Optional[Dict]:
    try:
        r = requests.post(
            f"{API_BASE}{endpoint}",
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error(f"Cannot reach API at {API_BASE}. Is uvicorn running?")
    except requests.exceptions.Timeout:
        st.error("API request timed out. The server may be overloaded — try again.")
    except requests.exceptions.HTTPError as e:
        # API may return HTML (nginx/uvicorn 500 page) instead of JSON on crashes
        try:
            detail = e.response.json().get("detail", str(e))
        except Exception:
            detail = f"HTTP {e.response.status_code} — server returned a non-JSON error body. Check uvicorn logs."
        st.error(f"API error {e.response.status_code}: {detail}")
    except Exception as e:
        st.error(f"Unexpected error: {e}")
    return None


@st.cache_data(ttl=300)
def fetch_health() -> Optional[Dict]:
    return _get("/health")


@st.cache_data(ttl=300)
def fetch_match_ids() -> List[int]:
    data = _get("/matches")
    return data["match_ids"] if data else []


# ---------------------------------------------------------------------------
# Sidebar — navigation + health indicator
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("🏏 Cricket ML")
    st.caption("IPL Prediction Suite")
    st.divider()

    page = st.radio(
        "Navigate",
        options=[
            "🏏 Match Winner",
            "📊 Score Predictor",
            "📈 Win Probability Curve",
            "🏆 POTM Predictor",
            "🔍 Player Stats",
        ],
        label_visibility="collapsed",
    )

    st.divider()
    health = fetch_health()
    if health:
        st.success("API connected")
        with st.expander("Model status"):
            for model, loaded in health.get("models_loaded", {}).items():
                icon = "✅" if loaded else "❌"
                st.write(f"{icon}  {model}")
        TEAMS = health.get("teams", [])
        VENUES = health.get("venues", [])
        if not TEAMS or not VENUES:
            st.warning(
                "⚠️ API returned no teams/venues — your `api/schemas.py` "
                "may be outdated. Replace it with the latest output and restart uvicorn."
            )
    else:
        st.error("API offline")
        TEAMS = []
        VENUES = []

    st.caption(f"API: `{API_BASE}`")


# ---------------------------------------------------------------------------
# Chart helper — defined at module scope so it is available inside all elif
# branches without breaking the if/elif chain.
# ---------------------------------------------------------------------------


def _build_win_curve_fig(
    overs: List[int],
    probs: List[float],
    batting: str,
    bowling: str,
    winner: str,
) -> go.Figure:
    fig = go.Figure()
    fig.add_hrect(y0=45, y1=55, fillcolor="gray", opacity=0.1, line_width=0)
    fig.add_hline(y=50, line_dash="dash", line_color="gray", line_width=1)
    fig.add_trace(
        go.Scatter(
            x=overs,
            y=probs,
            mode="lines+markers",
            name=f"{batting} win %",
            line=dict(color="#00b4d8", width=3),
            marker=dict(size=7),
            hovertemplate="Over %{x}<br>Win prob: %{y:.1f}%<extra></extra>",
        )
    )
    if overs:
        fig.add_annotation(
            x=overs[-1],
            y=probs[-1],
            text=f"  {probs[-1]:.1f}%",
            showarrow=False,
            font=dict(color="#00b4d8", size=13),
        )
    fig.update_layout(
        xaxis_title="Over",
        yaxis_title=f"{batting} win probability (%)",
        yaxis=dict(range=[0, 100]),
        xaxis=dict(range=[min(overs) - 0.5, max(overs) + 0.5] if overs else [0, 20]),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(t=30, b=40),
    )
    return fig


# ---------------------------------------------------------------------------
# Player Stats helpers — pure pandas, no API call needed
# All functions are @st.cache_data so re-selecting a player is instant.
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def _load_deliveries_local() -> pd.DataFrame:
    path = Path(__file__).parent / "data" / "processed" / "deliveries.parquet"
    df = pd.read_parquet(path)
    # Ensure derived columns exist regardless of parquet version
    if "is_legal_delivery" not in df.columns:
        df["is_legal_delivery"] = (df["wide_runs"] == 0) & (df["noball_runs"] == 0)
    if "is_wicket" not in df.columns:
        df["is_wicket"] = df["player_dismissed"].notna()
    return df


@st.cache_data(show_spinner=False)
def _load_matches_local() -> pd.DataFrame:
    path = Path(__file__).parent / "data" / "processed" / "matches.parquet"
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def _all_players(deliveries: pd.DataFrame) -> List[str]:
    batters = set(deliveries["batsman"].dropna().unique())
    bowlers = set(deliveries["bowler"].dropna().unique())
    return sorted(batters | bowlers)


@st.cache_data(show_spinner=False)
def _batting_career(
    deliveries: pd.DataFrame,
    matches: pd.DataFrame,
    player: str,
    season_range: Tuple[int, int],
) -> Tuple[dict, pd.DataFrame]:
    """Returns (career_dict, season_df). Both empty when no data."""
    match_season = matches[["id", "season"]].rename(columns={"id": "match_id"})

    df = deliveries[
        (deliveries["batsman"] == player) & (~deliveries["is_super_over"])
    ].merge(match_season, on="match_id", how="left")
    df = df[df["season"].between(season_range[0], season_range[1])]
    if df.empty:
        return {}, pd.DataFrame()

    # Dismissals in the filtered season range
    dis_df = deliveries[
        (deliveries["player_dismissed"] == player) & (~deliveries["is_super_over"])
    ].merge(match_season, on="match_id", how="left")
    dis_df = dis_df[dis_df["season"].between(season_range[0], season_range[1])]
    dismissals = len(dis_df)

    runs = int(df["batsman_runs"].sum())
    balls = int(df["is_legal_delivery"].sum())
    fours = int((df["batsman_runs"] == 4).sum())
    sixes = int((df["batsman_runs"] == 6).sum())
    dots = int((df["is_legal_delivery"] & (df["batsman_runs"] == 0)).sum())
    ones = int((df["batsman_runs"] == 1).sum())
    twos = int((df["batsman_runs"] == 2).sum())

    career = {
        "matches": int(df["match_id"].nunique()),
        "runs": runs,
        "balls": balls,
        "fours": fours,
        "sixes": sixes,
        "dismissals": dismissals,
        "strike_rate": round(runs / balls * 100, 2) if balls > 0 else 0.0,
        # Not-out: average = total runs (standard cricket convention)
        "average": round(runs / dismissals, 2) if dismissals > 0 else runs,
        "dot_pct": round(dots / balls * 100, 1) if balls > 0 else 0.0,
        "boundary_pct": round((fours + sixes) / balls * 100, 1) if balls > 0 else 0.0,
        # Breakdown for donut chart
        "dots": dots,
        "ones": ones,
        "twos": twos,
    }

    season_df = (
        df.groupby("season")
        .agg(
            runs=("batsman_runs", "sum"),
            balls=("is_legal_delivery", "sum"),
            matches=("match_id", "nunique"),
        )
        .reset_index()
        .assign(
            strike_rate=lambda x: (x["runs"] / x["balls"].clip(lower=1) * 100).round(1)
        )
    )
    return career, season_df


@st.cache_data(show_spinner=False)
def _bowling_career(
    deliveries: pd.DataFrame,
    matches: pd.DataFrame,
    player: str,
    season_range: Tuple[int, int],
) -> Tuple[dict, pd.DataFrame]:
    """Returns (career_dict, season_df)."""
    match_season = matches[["id", "season"]].rename(columns={"id": "match_id"})

    df = deliveries[
        (deliveries["bowler"] == player) & (~deliveries["is_super_over"])
    ].merge(match_season, on="match_id", how="left")
    df = df[df["season"].between(season_range[0], season_range[1])].copy()
    if df.empty:
        return {}, pd.DataFrame()

    # Runs conceded: strip byes + legbyes (they don't count against the bowler)
    bye_col = (
        df["bye_runs"] if "bye_runs" in df.columns else pd.Series(0, index=df.index)
    )
    leg_col = (
        df["legbye_runs"]
        if "legbye_runs" in df.columns
        else pd.Series(0, index=df.index)
    )
    df["_rc"] = df["total_runs"] - bye_col - leg_col

    # Wickets credited to the bowler (exclude run-outs etc.)
    NON_BOWLER = {"run out", "retired hurt", "obstructing the field"}
    if "dismissal_kind" in df.columns:
        df["_wkt"] = df["player_dismissed"].notna() & ~df["dismissal_kind"].isin(
            NON_BOWLER
        )
    else:
        df["_wkt"] = df["player_dismissed"].notna()

    runs_conceded = int(df["_rc"].sum())
    wickets = int(df["_wkt"].sum())
    legal = int(df["is_legal_delivery"].sum())
    dots = int((df["is_legal_delivery"] & (df["total_runs"] == 0)).sum())

    career = {
        "matches": int(df["match_id"].nunique()),
        "wickets": wickets,
        "runs_conceded": runs_conceded,
        "overs": round(legal / 6, 1),
        "economy": round(runs_conceded / (legal / 6), 2) if legal >= 6 else 0.0,
        "average": round(runs_conceded / wickets, 2) if wickets > 0 else float("inf"),
        "strike_rate": round(legal / wickets, 2) if wickets > 0 else float("inf"),
        "dot_pct": round(dots / legal * 100, 1) if legal > 0 else 0.0,
    }

    season_df = (
        df.groupby("season")
        .agg(
            wickets=("_wkt", "sum"),
            runs_conceded=("_rc", "sum"),
            legal=("is_legal_delivery", "sum"),
            matches=("match_id", "nunique"),
        )
        .reset_index()
        .assign(
            overs=lambda x: (x["legal"] / 6).round(1),
            economy=lambda x: (
                x["runs_conceded"] / (x["legal"] / 6).clip(lower=1)
            ).round(2),
        )
        .astype({"wickets": int, "matches": int})
    )
    return career, season_df


@st.cache_data(show_spinner=False)
def _leaderboard(deliveries: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """All-time top-15 batters and bowlers across all seasons."""
    d = deliveries[~deliveries["is_super_over"]]

    bat_lb = (
        d.groupby("batsman")
        .agg(
            runs=("batsman_runs", "sum"),
            balls=("is_legal_delivery", "sum"),
            fours=("batsman_runs", lambda x: (x == 4).sum()),
            sixes=("batsman_runs", lambda x: (x == 6).sum()),
            matches=("match_id", "nunique"),
        )
        .reset_index()
        .rename(columns={"batsman": "Player"})
        .assign(SR=lambda x: (x["runs"] / x["balls"].clip(1) * 100).round(1))
        .query("balls >= 50")  # minimum 50 balls to qualify
        .sort_values("runs", ascending=False)
        .head(15)
        .reset_index(drop=True)
    )
    bat_lb.index += 1

    NON_BOWLER = {"run out", "retired hurt", "obstructing the field"}
    bd = d.copy()
    if "dismissal_kind" in bd.columns:
        bd["_wkt"] = bd["player_dismissed"].notna() & ~bd["dismissal_kind"].isin(
            NON_BOWLER
        )
    else:
        bd["_wkt"] = bd["player_dismissed"].notna()

    bye_col = (
        bd["bye_runs"] if "bye_runs" in bd.columns else pd.Series(0, index=bd.index)
    )
    leg_col = (
        bd["legbye_runs"]
        if "legbye_runs" in bd.columns
        else pd.Series(0, index=bd.index)
    )
    bd["_rc"] = bd["total_runs"] - bye_col - leg_col

    bowl_lb = (
        bd.groupby("bowler")
        .agg(
            wickets=("_wkt", "sum"),
            runs_conceded=("_rc", "sum"),
            legal=("is_legal_delivery", "sum"),
            matches=("match_id", "nunique"),
        )
        .reset_index()
        .rename(columns={"bowler": "Player"})
        .assign(
            overs=lambda x: (x["legal"] / 6).round(1),
            economy=lambda x: (x["runs_conceded"] / (x["legal"] / 6).clip(1)).round(2),
        )
        .query("legal >= 60")  # minimum 10 overs to qualify
        .sort_values("wickets", ascending=False)
        .head(15)
        .reset_index(drop=True)
        .astype({"wickets": int, "matches": int})
    )
    bowl_lb.index += 1

    return bat_lb, bowl_lb


# ===========================================================================
# PAGE 1 — Match Winner
# ===========================================================================

if page == "🏏 Match Winner":
    st.title("🏏 Pre-Match Win Probability")
    st.caption("XGBoost classifier · features: toss, venue, rolling win rates")

    if not TEAMS or not VENUES:
        st.warning("Waiting for API…")
        st.stop()

    col_l, col_r = st.columns(2)
    with col_l:
        team1 = st.selectbox(
            "Team 1", TEAMS, index=TEAMS.index("MI") if "MI" in TEAMS else 0
        )
    with col_r:
        team2 = st.selectbox(
            "Team 2",
            [t for t in TEAMS if t != team1],
            index=0,
        )

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        venue = st.selectbox("Venue", VENUES)
    with col_b:
        toss_winner = st.radio(
            "Toss won by",
            ["team1", "team2"],
            format_func=lambda x: team1 if x == "team1" else team2,
        )
    with col_c:
        toss_decision = st.radio("Toss decision", ["bat", "field"])

    season = st.slider("Season", min_value=2008, max_value=2023, value=2019)

    if st.button("Predict", type="primary", use_container_width=True):
        with st.spinner("Calling API…"):
            result = _post(
                "/predict/match-winner",
                {
                    "team1": team1,
                    "team2": team2,
                    "venue": venue,
                    "toss_winner": toss_winner,
                    "toss_decision": toss_decision,
                    "season": season,
                },
            )

        if result:
            p1 = result["team1_win_probability"]
            p2 = result["team2_win_probability"]

            st.divider()
            st.subheader("Result")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown(
                    f'<div class="metric-card">'
                    f"<p>{team1}</p><h2>{p1:.1f}%</h2></div>",
                    unsafe_allow_html=True,
                )
            with col2:
                st.markdown(
                    f'<div class="metric-card">'
                    f"<p>{team2}</p><h2>{p2:.1f}%</h2></div>",
                    unsafe_allow_html=True,
                )

            # Win bar
            st.markdown(
                f'<div class="win-bar-container">'
                f'<div class="win-bar-t1" style="width:{p1}%">{team1} {p1:.0f}%</div>'
                f'<div class="win-bar-t2" style="width:{p2}%">{team2} {p2:.0f}%</div>'
                f"</div>",
                unsafe_allow_html=True,
            )

            # Gauge
            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=p1,
                    number={"suffix": "%", "font": {"size": 36}},
                    title={"text": f"{team1} win probability", "font": {"size": 16}},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#00b4d8"},
                        "steps": [
                            {"range": [0, 40], "color": "rgba(255, 107, 107, 0.2)"},
                            {"range": [60, 100], "color": "rgba(0, 180, 216, 0.2)"},
                        ],
                        "threshold": {
                            "line": {"color": "white", "width": 2},
                            "thickness": 0.75,
                            "value": 50,
                        },
                    },
                )
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=280,
                margin=dict(t=40, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)


# ===========================================================================
# PAGE 2 — Score Predictor
# ===========================================================================

elif page == "📊 Score Predictor":
    st.title("📊 1st Innings Score Predictor")

    if not TEAMS or not VENUES:
        st.warning("Waiting for API…")
        st.stop()

    tab_lgb, tab_gru = st.tabs(
        [
            "⚡ LightGBM  (over-10 snapshot)",
            "🧠 GRU  (rolling over-by-over)",
        ]
    )

    # ── Shared score range chart helper ───────────────────────────────────
    def _score_range_chart(low: float, pred: float, high: float) -> None:
        fig = go.Figure()
        fig.add_shape(
            type="line",
            x0=low,
            x1=high,
            y0=1,
            y1=1,
            line=dict(color="#555", width=4),
        )
        fig.add_trace(
            go.Scatter(
                x=[low, pred, high],
                y=[1, 1, 1],
                mode="markers",
                marker=dict(
                    size=[16, 28, 16],
                    color=["#aaa", "#00b4d8", "#aaa"],
                    symbol=["circle", "diamond", "circle"],
                ),
                hovertemplate="%{x:.0f} runs<extra></extra>",
            )
        )
        fig.update_layout(
            title=f"Predicted range: {low:.0f} – {high:.0f} runs",
            xaxis_title="Final score",
            yaxis=dict(visible=False),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=180,
            margin=dict(t=40, b=20),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    # =========================================================
    # TAB 1 — LightGBM (existing, unchanged)
    # =========================================================
    with tab_lgb:
        st.caption(
            "LightGBM regressor · single snapshot at end of over 10 · MAE ≈ 13 runs"
        )

        col_l, col_r = st.columns(2)
        with col_l:
            lgb_team = st.selectbox("Batting team", TEAMS, key="lgb_team")
            lgb_venue = st.selectbox("Venue", VENUES, key="lgb_venue")
            lgb_season = st.slider("Season", 2008, 2023, 2019, key="lgb_season")
        with col_r:
            runs_10 = st.number_input(
                "Runs after 10 overs",
                min_value=0,
                max_value=150,
                value=62,
                key="lgb_runs",
            )
            wickets_10 = st.number_input(
                "Wickets after 10 overs",
                min_value=0,
                max_value=10,
                value=2,
                key="lgb_wkts",
            )
            boundaries_10 = st.number_input(
                "Boundaries (4s+6s) in 10 overs", min_value=0, value=8, key="lgb_bdry"
            )

        if st.button(
            "Predict Final Score",
            type="primary",
            use_container_width=True,
            key="lgb_btn",
        ):
            with st.spinner("Calling API…"):
                result = _post(
                    "/predict/score",
                    {
                        "batting_team": lgb_team,
                        "venue": lgb_venue,
                        "season": lgb_season,
                        "runs_10": int(runs_10),
                        "wickets_10": int(wickets_10),
                        "boundaries_10": int(boundaries_10),
                    },
                )
            if result:
                st.divider()
                c1, c2, c3 = st.columns(3)
                c1.metric(
                    "Predicted Score",
                    f"{result['predicted_final_score']:.0f}",
                    delta=f"{result['predicted_final_score'] - result['projected_naive']:.0f} vs naive",
                )
                c2.metric("Current RR", f"{result['current_rr']:.2f}")
                c3.metric("Naive (RR×20)", f"{result['projected_naive']:.0f}")
                _score_range_chart(
                    result["confidence_interval_low"],
                    result["predicted_final_score"],
                    result["confidence_interval_high"],
                )
                st.info(
                    f"Model: **{result['predicted_final_score']:.0f}** runs  |  "
                    f"Naive: **{result['projected_naive']:.0f}** runs  |  "
                    f"Interval: ±13 runs (MAE-based)"
                )

    # =========================================================
    # TAB 2 — GRU (rolling over-by-over)
    # =========================================================
    with tab_gru:
        st.caption(
            "2-layer GRU · ingests each over as a timestep · "
            "updates prediction after every over · MAE ≈ 9 runs"
        )

        # ── Context inputs ─────────────────────────────────────────────────
        gc1, gc2, gc3 = st.columns(3)
        with gc1:
            gru_team = st.selectbox("Batting team", TEAMS, key="gru_team")
        with gc2:
            gru_venue = st.selectbox("Venue", VENUES, key="gru_venue")
        with gc3:
            gru_season = st.slider("Season", 2008, 2023, 2019, key="gru_season")

        st.divider()

        # ── Per-over entry table ───────────────────────────────────────────
        # Session state: list of overs entered so far
        if "gru_overs" not in st.session_state:
            st.session_state.gru_overs: List[Dict] = []

        n_done = len(st.session_state.gru_overs)
        st.subheader(f"Overs entered: {n_done} / 20")

        if n_done < 20:
            st.markdown(f"**Enter over {n_done + 1}:**")
            oc1, oc2, oc3, oc4 = st.columns(4)
            with oc1:
                ov_runs = st.number_input(
                    "Runs", min_value=0, max_value=36, value=7, key=f"ov_r_{n_done}"
                )
            with oc2:
                ov_wkts = st.number_input(
                    "Wickets", min_value=0, max_value=10, value=0, key=f"ov_w_{n_done}"
                )
            with oc3:
                ov_bdry = st.number_input(
                    "Boundaries",
                    min_value=0,
                    max_value=36,
                    value=1,
                    key=f"ov_b_{n_done}",
                )
            with oc4:
                st.write("")  # vertical spacer
                st.write("")
                if st.button("➕ Add over", key=f"add_{n_done}"):
                    st.session_state.gru_overs.append(
                        {
                            "runs_in_over": int(ov_runs),
                            "wickets_in_over": int(ov_wkts),
                            "boundaries_in_over": int(ov_bdry),
                        }
                    )
                    st.rerun()

        # ── Table of entered overs ─────────────────────────────────────────
        if st.session_state.gru_overs:
            cum_r, cum_w = 0, 0
            rows = []
            for i, ov in enumerate(st.session_state.gru_overs, start=1):
                cum_r += ov["runs_in_over"]
                cum_w += ov["wickets_in_over"]
                rows.append(
                    {
                        "Over": i,
                        "Runs": ov["runs_in_over"],
                        "Wkts": ov["wickets_in_over"],
                        "Boundaries": ov["boundaries_in_over"],
                        "Cum Runs": cum_r,
                        "Cum Wkts": cum_w,
                        "RR": round(cum_r / i, 2),
                    }
                )
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

            bc1, bc2 = st.columns([1, 1])
            with bc1:
                if st.button("🗑 Remove last over", key="gru_remove"):
                    st.session_state.gru_overs.pop()
                    st.rerun()
            with bc2:
                if st.button("🔄 Reset all overs", key="gru_reset"):
                    st.session_state.gru_overs = []
                    st.rerun()

            st.divider()

            # ── Predict button ─────────────────────────────────────────────
            if st.button(
                "🧠 Predict Final Score (GRU)",
                type="primary",
                use_container_width=True,
                key="gru_btn",
            ):
                payload = {
                    "batting_team": gru_team,
                    "venue": gru_venue,
                    "season": gru_season,
                    "overs": st.session_state.gru_overs,
                }
                with st.spinner("Running GRU inference…"):
                    result = _post("/predict/score/gru", payload)

                if result:
                    st.divider()
                    m1, m2, m3 = st.columns(3)
                    m1.metric(
                        "GRU Predicted Score",
                        f"{result['predicted_final_score']:.0f}",
                    )
                    overs_in = result["overs_seen"]
                    cum_runs_now = sum(
                        o["runs_in_over"] for o in st.session_state.gru_overs
                    )
                    naive_proj = (
                        round(cum_runs_now / overs_in * 20) if overs_in > 0 else 0
                    )
                    m2.metric("Overs entered", overs_in)
                    m3.metric("Naive projection", naive_proj)

                    _score_range_chart(
                        result["confidence_interval_low"],
                        result["predicted_final_score"],
                        result["confidence_interval_high"],
                    )

                    st.info(
                        f"GRU has seen **{overs_in}** overs · "
                        f"Predicted: **{result['predicted_final_score']:.0f}** runs · "
                        f"Interval: ±{result['predicted_final_score'] - result['confidence_interval_low']:.0f} runs (val MAE)"
                    )

                    # ── Live prediction curve (call API after each over) ───
                    if n_done >= 3:
                        st.subheader("📈 Prediction as innings progressed")
                        curve_preds, curve_overs = [], []
                        for k in range(1, n_done + 1):
                            sub_payload = {
                                "batting_team": gru_team,
                                "venue": gru_venue,
                                "season": gru_season,
                                "overs": st.session_state.gru_overs[:k],
                            }
                            sub_res = _post("/predict/score/gru", sub_payload)
                            if sub_res:
                                curve_overs.append(k)
                                curve_preds.append(sub_res["predicted_final_score"])

                        if curve_overs:
                            fig_curve = go.Figure()
                            fig_curve.add_trace(
                                go.Scatter(
                                    x=curve_overs,
                                    y=curve_preds,
                                    mode="lines+markers",
                                    name="GRU prediction",
                                    line=dict(color="#00b4d8", width=2),
                                    marker=dict(size=7),
                                    hovertemplate="After over %{x}: %{y:.0f} runs<extra></extra>",
                                )
                            )
                            # Naive projection per over for comparison
                            naive_curve = [
                                round(
                                    sum(
                                        st.session_state.gru_overs[i]["runs_in_over"]
                                        for i in range(k)
                                    )
                                    / k
                                    * 20
                                )
                                for k in curve_overs
                            ]
                            fig_curve.add_trace(
                                go.Scatter(
                                    x=curve_overs,
                                    y=naive_curve,
                                    mode="lines",
                                    name="Naive (RR×20)",
                                    line=dict(color="#ffd700", width=1, dash="dot"),
                                    hovertemplate="Naive after over %{x}: %{y:.0f}<extra></extra>",
                                )
                            )
                            fig_curve.update_layout(
                                xaxis_title="Overs completed",
                                yaxis_title="Predicted final score",
                                legend=dict(orientation="h", y=1.12),
                                paper_bgcolor="rgba(0,0,0,0)",
                                plot_bgcolor="rgba(0,0,0,0)",
                                height=350,
                                margin=dict(t=30, b=40),
                            )
                            st.plotly_chart(fig_curve, use_container_width=True)
        else:
            st.info("Enter at least one over above to start predicting.")


# ===========================================================================
# PAGE 3 — Win Probability Curve
# ===========================================================================

elif page == "📈 Win Probability Curve":
    st.title("📈 Live Win Probability — Over by Over")
    st.caption("LightGBM · AUC 0.864 · 2nd innings state features")

    match_ids = fetch_match_ids()
    if not match_ids:
        st.warning("No match IDs available from API.")
        st.stop()

    col_l, col_r = st.columns([2, 1])
    with col_l:
        selected_id = st.selectbox(
            "Select match ID",
            match_ids,
            index=0,
            format_func=lambda x: f"Match #{x}",
        )
    with col_r:
        animate = st.checkbox("Animate over-by-over", value=True)

    if st.button("Load Win Curve", type="primary", use_container_width=True):
        with st.spinner("Fetching win curve…"):
            result = _get(f"/predict/win-curve/{selected_id}")

        if result:
            batting = result["batting_team"]
            bowling = result["bowling_team"]
            winner = result.get("actual_winner", "Unknown")
            curve = result["curve"]

            overs = [e["over"] for e in curve]
            probs = [e["win_probability"] for e in curve]

            st.subheader(f"Match #{selected_id}: {batting} vs {bowling}")
            st.caption(f"Actual winner: **{winner}**")

            if animate and len(overs) > 1:
                chart_placeholder = st.empty()
                for i in range(1, len(overs) + 1):
                    fig = _build_win_curve_fig(
                        overs[:i], probs[:i], batting, bowling, winner
                    )
                    chart_placeholder.plotly_chart(fig, use_container_width=True)
                    time.sleep(0.25)
            else:
                fig = _build_win_curve_fig(overs, probs, batting, bowling, winner)
                st.plotly_chart(fig, use_container_width=True)

            # Summary table
            df_curve = pd.DataFrame({"Over": overs, f"{batting} Win %": probs})
            df_curve["Phase"] = pd.cut(
                df_curve["Over"],
                bins=[0, 6, 15, 20],
                labels=["Powerplay", "Middle", "Death"],
                right=True,
            )
            st.dataframe(
                df_curve.style.background_gradient(
                    subset=[f"{batting} Win %"],
                    cmap="RdYlGn",
                    vmin=0,
                    vmax=100,
                ),
                width="stretch",
                hide_index=True,
            )

# ===========================================================================
# PAGE 4 — POTM Predictor
# ===========================================================================

elif page == "🏆 POTM Predictor":
    st.title("🏆 Player of the Match Predictor")
    st.caption("XGBoost classifier · imbalanced class with scale_pos_weight")

    st.info(
        "Enter the performance stats for each player. Add rows for all participants."
    )

    if "potm_players" not in st.session_state:
        st.session_state.potm_players = [
            {
                "player_name": "Rohit Sharma",
                "runs_scored": 78,
                "balls_faced": 48,
                "wickets_taken": 0,
                "runs_given": 0,
                "balls_bowled": 0,
                "player_won": 1,
            },
            {
                "player_name": "Jasprit Bumrah",
                "runs_scored": 4,
                "balls_faced": 3,
                "wickets_taken": 4,
                "runs_given": 18,
                "balls_bowled": 24,
                "player_won": 1,
            },
        ]

    # ── Add / remove rows ─────────────────────────────────────────────────
    col_add, col_clear = st.columns([1, 1])
    with col_add:
        if st.button("➕ Add player"):
            st.session_state.potm_players.append(
                {
                    "player_name": f"Player {len(st.session_state.potm_players) + 1}",
                    "runs_scored": 0,
                    "balls_faced": 0,
                    "wickets_taken": 0,
                    "runs_given": 0,
                    "balls_bowled": 0,
                    "player_won": 0,
                }
            )
    with col_clear:
        if st.button("🗑️ Clear all"):
            st.session_state.potm_players = []

    # ── Player input rows ─────────────────────────────────────────────────
    to_delete = []
    for i, p in enumerate(st.session_state.potm_players):
        with st.expander(f"Player {i+1}: {p['player_name']}", expanded=True):
            cols = st.columns([2, 1, 1, 1, 1, 1, 1, 0.4])
            p["player_name"] = cols[0].text_input(
                "Name", value=p["player_name"], key=f"name_{i}"
            )
            p["runs_scored"] = cols[1].number_input(
                "Runs", value=p["runs_scored"], min_value=0, key=f"runs_{i}"
            )
            p["balls_faced"] = cols[2].number_input(
                "Balls (bat)", value=p["balls_faced"], min_value=0, key=f"bf_{i}"
            )
            p["wickets_taken"] = cols[3].number_input(
                "Wkts",
                value=p["wickets_taken"],
                min_value=0,
                max_value=10,
                key=f"wk_{i}",
            )
            p["runs_given"] = cols[4].number_input(
                "Runs given", value=p["runs_given"], min_value=0, key=f"rg_{i}"
            )
            p["balls_bowled"] = cols[5].number_input(
                "Balls (bowl)", value=p["balls_bowled"], min_value=0, key=f"bb_{i}"
            )
            p["player_won"] = int(
                cols[6].checkbox("Won?", value=bool(p["player_won"]), key=f"pw_{i}")
            )
            if cols[7].button("✕", key=f"del_{i}"):
                to_delete.append(i)

    for i in reversed(to_delete):
        st.session_state.potm_players.pop(i)

    # ── Predict ───────────────────────────────────────────────────────────
    if st.session_state.potm_players and st.button(
        "Predict POTM", type="primary", use_container_width=True
    ):
        payload = {"players": st.session_state.potm_players}
        with st.spinner("Calling API…"):
            result = _post("/predict/potm", payload)

        if result:
            st.divider()
            st.subheader(f"🏆 Predicted POTM: **{result['predicted_potm']}**")

            players_sorted = result["players"]  # already sorted by rank
            df = pd.DataFrame(
                [
                    {
                        "Rank": p["rank"],
                        "Player": p["player_name"],
                        "POTM Probability": f"{p['potm_probability']:.1f}%",
                        # Uniformly string — prevents PyArrow mixed-type serialisation crash
                        "Strike Rate": (
                            f"{p['strike_rate']:.2f}" if p["strike_rate"] > 0 else "—"
                        ),
                        "Economy": f"{p['economy']:.2f}" if p["economy"] > 0 else "—",
                    }
                    for p in players_sorted
                ]
            )

            st.dataframe(df, width="stretch", hide_index=True)

            # Horizontal bar chart
            names = [p["player_name"] for p in players_sorted]
            probs = [p["potm_probability"] for p in players_sorted]
            colors = ["#ffd700" if i == 0 else "#00b4d8" for i in range(len(names))]

            fig = go.Figure(
                go.Bar(
                    x=probs,
                    y=names,
                    orientation="h",
                    marker_color=colors,
                    text=[f"{p:.1f}%" for p in probs],
                    textposition="outside",
                    hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
                )
            )
            fig.update_layout(
                xaxis_title="POTM probability (%)",
                xaxis=dict(range=[0, min(100, max(probs) * 1.3)]),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=max(280, len(names) * 48),
                margin=dict(t=20, b=20, l=140),
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)


# ===========================================================================
# PAGE 5 — Player Stats Explorer
# ===========================================================================

elif page == "🔍 Player Stats":
    st.title("🔍 Player Stats Explorer")
    st.caption("Ball-by-ball aggregation from deliveries data · no ML model involved")

    # ── Load processed parquets directly (no API round-trip needed) ───────
    with st.spinner("Loading ball-by-ball data…"):
        _deliv = _load_deliveries_local()
        _matches = _load_matches_local()

    all_players = _all_players(_deliv)
    all_seasons = sorted(_matches["season"].dropna().unique().tolist())
    season_min = int(min(all_seasons))
    season_max = int(max(all_seasons))

    # ── Controls ──────────────────────────────────────────────────────────
    col_search, col_season = st.columns([2, 1])
    with col_search:
        default_idx = all_players.index("MS Dhoni") if "MS Dhoni" in all_players else 0
        player = st.selectbox("Select player", all_players, index=default_idx)
    with col_season:
        season_range = st.slider(
            "Season range",
            min_value=season_min,
            max_value=season_max,
            value=(season_min, season_max),
        )

    tab_bat, tab_bowl, tab_lb = st.tabs(["🏏 Batting", "🎯 Bowling", "🏆 Leaderboard"])

    # ── TAB 1 · Batting ───────────────────────────────────────────────────
    with tab_bat:
        career, season_df = _batting_career(_deliv, _matches, player, season_range)

        if not career:
            st.info(
                f"No batting records for **{player}** in seasons {season_range[0]}–{season_range[1]}."
            )
        else:
            # Career metric row
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            c1.metric("Matches", career["matches"])
            c2.metric("Runs", career["runs"])
            avg_label = (
                f"{career['average']:.1f}"
                if isinstance(career["average"], float)
                and career["average"] != float("inf")
                else str(career["average"])
            )
            c3.metric("Average", avg_label)
            c4.metric("Strike Rate", f"{career['strike_rate']:.1f}")
            c5.metric("4s", career["fours"])
            c6.metric("6s", career["sixes"])
            st.caption(
                f"Balls faced: **{career['balls']}** · "
                f"Dot ball %: **{career['dot_pct']}%** · "
                f"Boundary %: **{career['boundary_pct']}%** · "
                f"Dismissals: **{career['dismissals']}**"
            )

            # Season chart: Runs bars + Strike Rate line (dual y-axis)
            if not season_df.empty:
                st.subheader("Season by season")
                fig_bat = go.Figure()
                fig_bat.add_trace(
                    go.Bar(
                        x=season_df["season"],
                        y=season_df["runs"],
                        name="Runs",
                        marker_color="#00b4d8",
                        yaxis="y1",
                        hovertemplate="Season %{x}<br>Runs: %{y}<extra></extra>",
                    )
                )
                fig_bat.add_trace(
                    go.Scatter(
                        x=season_df["season"],
                        y=season_df["strike_rate"],
                        name="Strike Rate",
                        mode="lines+markers",
                        line=dict(color="#ffd700", width=2),
                        marker=dict(size=8),
                        yaxis="y2",
                        hovertemplate="Season %{x}<br>SR: %{y}<extra></extra>",
                    )
                )
                fig_bat.update_layout(
                    xaxis=dict(
                        tickmode="array",
                        tickvals=season_df["season"].tolist(),
                        tickformat="d",
                    ),
                    yaxis=dict(title="Runs", side="left"),
                    yaxis2=dict(
                        title="Strike Rate",
                        overlaying="y",
                        side="right",
                        showgrid=False,
                    ),
                    legend=dict(orientation="h", y=1.12),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=380,
                    margin=dict(t=40, b=40),
                )
                st.plotly_chart(fig_bat, use_container_width=True)

            # Scoring breakdown donut
            st.subheader("Scoring breakdown")
            donut = go.Figure(
                go.Pie(
                    labels=["Dot balls", "1s", "2s", "4s", "6s"],
                    values=[
                        career["dots"],
                        career["ones"],
                        career["twos"],
                        career["fours"],
                        career["sixes"],
                    ],
                    hole=0.52,
                    marker_colors=["#444", "#90e0ef", "#00b4d8", "#0077b6", "#ffd700"],
                    textinfo="label+percent",
                    hovertemplate="%{label}: %{value} balls (%{percent})<extra></extra>",
                )
            )
            donut.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                height=340,
                margin=dict(t=20, b=10),
                legend=dict(orientation="h", y=-0.1),
            )
            st.plotly_chart(donut, use_container_width=True)

    # ── TAB 2 · Bowling ───────────────────────────────────────────────────
    with tab_bowl:
        bcareer, bseason_df = _bowling_career(_deliv, _matches, player, season_range)

        if not bcareer:
            st.info(
                f"No bowling records for **{player}** in seasons {season_range[0]}–{season_range[1]}."
            )
        else:
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Matches", bcareer["matches"])
            c2.metric("Wickets", bcareer["wickets"])
            c3.metric("Economy", f"{bcareer['economy']:.2f}")
            c4.metric(
                "Average",
                (
                    f"{bcareer['average']:.1f}"
                    if bcareer["average"] != float("inf")
                    else "—"
                ),
            )
            c5.metric(
                "Bowl SR",
                (
                    f"{bcareer['strike_rate']:.1f}"
                    if bcareer["strike_rate"] != float("inf")
                    else "—"
                ),
            )
            st.caption(
                f"Overs bowled: **{bcareer['overs']}** · "
                f"Runs conceded: **{bcareer['runs_conceded']}** · "
                f"Dot ball %: **{bcareer['dot_pct']}%**"
            )

            # Season chart: Wickets bars + Economy line (dual y-axis)
            if not bseason_df.empty:
                st.subheader("Season by season")
                fig_bowl = go.Figure()
                fig_bowl.add_trace(
                    go.Bar(
                        x=bseason_df["season"],
                        y=bseason_df["wickets"],
                        name="Wickets",
                        marker_color="#ff6b6b",
                        yaxis="y1",
                        hovertemplate="Season %{x}<br>Wickets: %{y}<extra></extra>",
                    )
                )
                fig_bowl.add_trace(
                    go.Scatter(
                        x=bseason_df["season"],
                        y=bseason_df["economy"],
                        name="Economy",
                        mode="lines+markers",
                        line=dict(color="#ffd700", width=2),
                        marker=dict(size=8),
                        yaxis="y2",
                        hovertemplate="Season %{x}<br>Economy: %{y}<extra></extra>",
                    )
                )
                fig_bowl.update_layout(
                    xaxis=dict(
                        tickmode="array",
                        tickvals=bseason_df["season"].tolist(),
                        tickformat="d",
                    ),
                    yaxis=dict(title="Wickets", side="left"),
                    yaxis2=dict(
                        title="Economy", overlaying="y", side="right", showgrid=False
                    ),
                    legend=dict(orientation="h", y=1.12),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=380,
                    margin=dict(t=40, b=40),
                )
                st.plotly_chart(fig_bowl, use_container_width=True)

    # ── TAB 3 · Leaderboard ───────────────────────────────────────────────
    with tab_lb:
        bat_lb, bowl_lb = _leaderboard(_deliv)

        st.subheader("🏏 All-time top run-scorers")
        st.caption("Minimum 50 balls faced to qualify")
        st.dataframe(
            bat_lb[["Player", "matches", "runs", "SR", "fours", "sixes"]].rename(
                columns={
                    "matches": "Matches",
                    "runs": "Runs",
                    "fours": "4s",
                    "sixes": "6s",
                }
            ),
            width="stretch",
            hide_index=False,
        )

        st.divider()
        st.subheader("🎯 All-time top wicket-takers")
        st.caption("Minimum 10 overs bowled to qualify")
        st.dataframe(
            bowl_lb[["Player", "matches", "wickets", "overs", "economy"]].rename(
                columns={
                    "matches": "Matches",
                    "wickets": "Wickets",
                    "overs": "Overs",
                    "economy": "Economy",
                }
            ),
            width="stretch",
            hide_index=False,
        )
