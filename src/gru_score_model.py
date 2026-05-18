"""
gru_score_model.py
------------------
PyTorch 2-layer GRU that predicts first-innings final score from an
over-by-over sequence of in-match features.

Architecture
------------
Each training example is a prefix of a match's first innings:
  (overs_1..K, static_context) → predicted_final_score

The same match of N completed overs generates N training examples
(one per prefix length), giving ~11k+ examples from 636 matches —
far more signal than the 636-row static LightGBM snapshot.

At inference time, pass whatever overs have been completed so far
(K = 1..20) and receive a score prediction.

Input per timestep (7 features):
    runs_in_over, wickets_in_over, cum_runs, cum_wickets,
    current_rr, boundaries_in_over, balls_in_over

Static context (3 features, concatenated at every timestep):
    batting_team_enc, venue_enc, season_norm

Total input dim per step: 10

Model: Linear(10→32) → GRU(32, hidden=64, layers=2, dropout=0.3) → Linear(64→1)

Saved as: models/gru_score_predictor.pt
  {
    "state_dict": model.state_dict(),
    "config":     {input_dim, hidden_dim, num_layers, dropout},
    "norm":       {mean, std dicts for all numeric features},
    "enc_maps":   {batting_team_enc_map, venue_enc_map},
    "meta":       {r2, mae, epochs_trained, sklearn_version},
  }
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Features computed per over (order matters — must match _over_features())
STEP_FEATURES: List[str] = [
    "runs_in_over",
    "wickets_in_over",
    "cum_runs",
    "cum_wickets",
    "current_rr",
    "boundaries_in_over",
    "balls_in_over",
]

# Static context features per match (appended at every timestep)
STATIC_FEATURES: List[str] = [
    "batting_team_enc",
    "venue_enc",
    "season_norm",
]

INPUT_DIM: int = len(STEP_FEATURES) + len(STATIC_FEATURES)  # 10


# ---------------------------------------------------------------------------
# Feature engineering — over-level aggregation
# ---------------------------------------------------------------------------


def build_over_sequences(
    deliveries: pd.DataFrame,
    matches: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate ball-by-ball deliveries into per-over features for 1st innings.

    Returns a DataFrame with one row per (match_id, over), columns:
        match_id, over, runs_in_over, wickets_in_over, cum_runs,
        cum_wickets, current_rr, boundaries_in_over, balls_in_over,
        batting_team, venue, season, final_score

    Assumes 1-indexed overs (1–20) as validated by data_loader.load_deliveries.
    Super-overs are excluded.
    """
    is_first = (deliveries["inning"] == 1) & (~deliveries["is_super_over"].astype(bool))
    d1 = deliveries[is_first].copy()

    # Per-over aggregation
    over_agg = (
        d1.groupby(["match_id", "over"])
        .agg(
            runs_in_over=("total_runs", "sum"),
            wickets_in_over=("is_wicket", "sum"),
            boundaries_in_over=("batsman_runs", lambda x: ((x == 4) | (x == 6)).sum()),
            balls_in_over=("is_legal_delivery", "sum"),
            batting_team=("batting_team", "first"),
        )
        .reset_index()
        .sort_values(["match_id", "over"])
    )

    # Cumulative stats within each match (running totals at end of each over)
    over_agg["cum_runs"] = over_agg.groupby("match_id")["runs_in_over"].cumsum()
    over_agg["cum_wickets"] = over_agg.groupby("match_id")["wickets_in_over"].cumsum()
    over_agg["current_rr"] = (over_agg["cum_runs"] / over_agg["over"]).round(3)

    # Final score per match (used as regression target)
    final_scores = (
        d1.groupby("match_id")["total_runs"].sum().reset_index(name="final_score")
    )

    # Match metadata
    match_meta = matches[["id", "venue", "season"]].rename(columns={"id": "match_id"})
    over_agg = over_agg.merge(final_scores, on="match_id", how="left").merge(
        match_meta, on="match_id", how="left"
    )

    logger.info(
        "build_over_sequences: %d rows, %d matches, %d unique overs",
        len(over_agg),
        over_agg["match_id"].nunique(),
        over_agg["over"].nunique(),
    )
    return over_agg.dropna()


# ---------------------------------------------------------------------------
# Encoding maps (reproduce cat.codes alphabetical ordering)
# ---------------------------------------------------------------------------


def build_enc_maps(over_df: pd.DataFrame) -> Dict[str, Dict[str, int]]:
    """Build {label: code} maps for batting_team and venue."""

    def _enc(series: pd.Series) -> Dict[str, int]:
        vals = sorted(series.dropna().unique().tolist())
        return {v: i for i, v in enumerate(vals)}

    return {
        "batting_team": _enc(over_df["batting_team"]),
        "venue": _enc(over_df["venue"]),
    }


# ---------------------------------------------------------------------------
# Normalisation (fit on train, apply on train+val+test)
# ---------------------------------------------------------------------------


def fit_normaliser(
    df: pd.DataFrame, features: List[str]
) -> Dict[str, Dict[str, float]]:
    """Return {feature: {mean, std}} computed on df."""
    stats: Dict[str, Dict[str, float]] = {}
    for f in features:
        mu = float(df[f].mean())
        std = float(df[f].std())
        if std < 1e-6:
            std = 1.0
        stats[f] = {"mean": mu, "std": std}
    return stats


def normalise(values: np.ndarray, mean: float, std: float) -> np.ndarray:
    return (values - mean) / std


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------


class OverSequenceDataset(Dataset):
    """
    One example per (match, prefix_length):
        X: float32 tensor of shape (prefix_len, INPUT_DIM)
        y: float32 scalar — final_score

    Variable-length sequences are returned as-is; collation with padding
    is handled by collate_fn (see make_dataloader).
    """

    def __init__(
        self,
        over_df: pd.DataFrame,
        enc_maps: Dict[str, Dict[str, int]],
        norm_stats: Dict[str, Dict[str, float]],
        season_range: Tuple[float, float],
        max_prefix: int = 20,
    ):
        self.samples: List[Tuple[torch.Tensor, float]] = []
        season_min, season_max = season_range

        for match_id, grp in over_df.groupby("match_id"):
            grp = grp.sort_values("over")
            n_overs = len(grp)
            final_score = float(grp["final_score"].iloc[0])

            bat_enc = enc_maps["batting_team"].get(grp["batting_team"].iloc[0], 0)
            venue_enc = enc_maps["venue"].get(grp["venue"].iloc[0], 0)
            season_raw = float(grp["season"].iloc[0])
            season_norm = (season_raw - season_min) / max(season_max - season_min, 1)

            static = np.array([bat_enc, venue_enc, season_norm], dtype=np.float32)

            # Build step feature matrix for all overs
            step_mat = np.zeros((n_overs, len(STEP_FEATURES)), dtype=np.float32)
            for col_idx, feat in enumerate(STEP_FEATURES):
                raw = grp[feat].values.astype(np.float32)
                if feat in norm_stats:
                    raw = normalise(
                        raw, norm_stats[feat]["mean"], norm_stats[feat]["std"]
                    ).astype(np.float32)
                step_mat[:, col_idx] = raw

            # Broadcast static context across all timesteps
            static_mat = np.tile(static, (n_overs, 1))  # (n_overs, 3)
            full_mat = np.concatenate([step_mat, static_mat], axis=1)  # (n_overs, 10)

            # One sample per prefix length (1..n_overs, capped at max_prefix)
            for k in range(1, min(n_overs, max_prefix) + 1):
                x_tensor = torch.from_numpy(full_mat[:k])  # (k, INPUT_DIM)
                self.samples.append((x_tensor, final_score))

        logger.info("OverSequenceDataset: %d samples", len(self.samples))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x, y = self.samples[idx]
        return x, torch.tensor(y, dtype=torch.float32)


def collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Pad variable-length sequences to the longest in the batch.
    Returns (padded_x, lengths, y).
    """
    xs, ys = zip(*batch)
    lengths = torch.tensor([x.shape[0] for x in xs], dtype=torch.long)
    padded = nn.utils.rnn.pad_sequence(xs, batch_first=True)  # (B, T_max, D)
    return padded, lengths, torch.stack(ys)


def make_dataloader(
    dataset: OverSequenceDataset,
    batch_size: int = 32,
    shuffle: bool = True,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=0,  # Windows: num_workers > 0 requires if __name__ == "__main__"
        pin_memory=False,
    )


# ---------------------------------------------------------------------------
# GRU Model
# ---------------------------------------------------------------------------


class GRUScorePredictor(nn.Module):
    """
    2-layer GRU for first-innings score regression.

    Architecture:
        Linear(INPUT_DIM → proj_dim) → ReLU
        GRU(proj_dim, hidden_dim, num_layers, dropout, batch_first)
        Linear(hidden_dim → 1)

    Only the last hidden state is used for prediction (many-to-one).
    Variable-length sequences are handled via pack_padded_sequence.
    """

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        proj_dim: int = 32,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.proj = nn.Sequential(
            nn.Linear(input_dim, proj_dim),
            nn.ReLU(),
        )
        self.gru = nn.GRU(
            input_size=proj_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        x: torch.Tensor,  # (B, T, INPUT_DIM)
        lengths: torch.Tensor,  # (B,) — actual sequence lengths
    ) -> torch.Tensor:  # (B,)
        projected = self.proj(x)  # (B, T, proj_dim)

        packed = pack_padded_sequence(
            projected, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h_n = self.gru(packed)  # h_n: (num_layers, B, hidden_dim)

        last_hidden = h_n[-1]  # (B, hidden_dim) — top layer's final state
        out = self.head(last_hidden).squeeze(-1)  # (B,)
        return out


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train_gru(
    train_ds: OverSequenceDataset,
    val_ds: OverSequenceDataset,
    hidden_dim: int = 64,
    num_layers: int = 2,
    dropout: float = 0.3,
    lr: float = 1e-3,
    epochs: int = 200,
    batch_size: int = 32,
    patience: int = 20,
    device: Optional[torch.device] = None,
) -> Tuple[GRUScorePredictor, Dict]:
    """
    Train the GRU score predictor with early stopping.

    Returns (best_model, metrics_dict).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training on device: %s", device)

    model = GRUScorePredictor(
        input_dim=INPUT_DIM,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", factor=0.5, patience=10
    )
    criterion = nn.MSELoss()

    train_loader = make_dataloader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = make_dataloader(val_ds, batch_size=batch_size, shuffle=False)

    best_val_loss = float("inf")
    best_state = None
    no_improve = 0
    history: List[Dict] = []

    for epoch in range(1, epochs + 1):
        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        train_losses: List[float] = []
        for x_batch, lengths, y_batch in train_loader:
            x_batch = x_batch.to(device)
            lengths = lengths.to(device)
            y_batch = y_batch.to(device)

            optimiser.zero_grad()
            preds = model(x_batch, lengths)
            loss = criterion(preds, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimiser.step()
            train_losses.append(loss.item())

        # ── Validate ───────────────────────────────────────────────────────
        model.eval()
        val_losses: List[float] = []
        all_preds, all_targets = [], []
        with torch.no_grad():
            for x_batch, lengths, y_batch in val_loader:
                x_batch = x_batch.to(device)
                lengths = lengths.to(device)
                preds = model(x_batch, lengths)
                val_losses.append(criterion(preds, y_batch.to(device)).item())
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(y_batch.numpy())

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        scheduler.step(val_loss)

        # MAE on validation set
        p_arr = np.array(all_preds)
        t_arr = np.array(all_targets)
        val_mae = float(np.mean(np.abs(p_arr - t_arr)))

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_mae": val_mae,
            }
        )

        if epoch % 20 == 0 or epoch == 1:
            logger.info(
                "Epoch %3d/%d  train_loss=%.2f  val_loss=%.2f  val_mae=%.2f",
                epoch,
                epochs,
                train_loss,
                val_loss,
                val_mae,
            )

        # Early stopping
        if val_loss < best_val_loss - 0.1:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info("Early stopping at epoch %d (patience=%d)", epoch, patience)
                break

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)

    # Final metrics
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for x_batch, lengths, y_batch in val_loader:
            preds = model(x_batch.to(device), lengths.to(device))
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y_batch.numpy())

    p_arr = np.array(all_preds)
    t_arr = np.array(all_targets)
    mae = float(np.mean(np.abs(p_arr - t_arr)))
    ss_res = float(np.sum((p_arr - t_arr) ** 2))
    ss_tot = float(np.sum((t_arr - t_arr.mean()) ** 2))
    r2 = round(1 - ss_res / ss_tot, 4) if ss_tot > 0 else 0.0

    metrics = {
        "val_mae": round(mae, 2),
        "val_r2": r2,
        "best_val_loss": round(best_val_loss, 4),
        "epochs_trained": len(history),
        "history": history,
    }
    logger.info("GRU final — MAE: %.2f | R²: %.4f", mae, r2)
    return model, metrics


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------


def save_gru(
    model: GRUScorePredictor,
    norm_stats: Dict,
    enc_maps: Dict,
    season_range: Tuple[float, float],
    metrics: Dict,
    path: str,
) -> None:
    """Save model weights + all inference artefacts to a single .pt file."""
    config = {
        "input_dim": INPUT_DIM,
        "proj_dim": model.proj[0].out_features,
        "hidden_dim": model.hidden_dim,
        "num_layers": model.num_layers,
        "dropout": model.gru.dropout,
    }
    payload = {
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "config": config,
        "norm_stats": norm_stats,
        "enc_maps": enc_maps,
        "season_range": season_range,
        "step_features": STEP_FEATURES,
        "static_features": STATIC_FEATURES,
        "meta": {
            "val_mae": metrics.get("val_mae"),
            "val_r2": metrics.get("val_r2"),
            "epochs_trained": metrics.get("epochs_trained"),
        },
    }
    torch.save(payload, path)
    logger.info("GRU model saved → %s", path)


def load_gru(
    path: str, device: Optional[torch.device] = None
) -> Tuple[GRUScorePredictor, Dict]:
    """
    Load a saved GRU model.

    Returns (model_on_device, payload_dict).
    The payload dict contains norm_stats, enc_maps, season_range, meta.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    payload = torch.load(path, map_location=device, weights_only=False)
    cfg = payload["config"]

    model = GRUScorePredictor(
        input_dim=cfg["input_dim"],
        proj_dim=cfg["proj_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
    )
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model, payload


# ---------------------------------------------------------------------------
# Inference helper — used by the API endpoint
# ---------------------------------------------------------------------------


def predict_from_overs(
    model: GRUScorePredictor,
    payload: Dict,
    over_rows: List[Dict],
    batting_team: str,
    venue: str,
    season: int,
    device: Optional[torch.device] = None,
) -> float:
    """
    Predict final score given a list of completed overs.

    over_rows: list of dicts, one per completed over, keys:
        runs_in_over, wickets_in_over, cum_runs, cum_wickets,
        current_rr, boundaries_in_over, balls_in_over

    Returns predicted final score (float).
    """
    if device is None:
        device = next(model.parameters()).device

    norm_stats = payload["norm_stats"]
    enc_maps = payload["enc_maps"]
    season_range = payload["season_range"]
    step_feats = payload.get("step_features", STEP_FEATURES)

    bat_enc = enc_maps["batting_team"].get(batting_team, 0)
    venue_enc = enc_maps["venue"].get(venue, 0)
    season_min, season_max = season_range
    season_norm = (season - season_min) / max(season_max - season_min, 1)
    static = np.array([bat_enc, venue_enc, season_norm], dtype=np.float32)

    n = len(over_rows)
    step_mat = np.zeros((n, len(step_feats)), dtype=np.float32)
    for col_idx, feat in enumerate(step_feats):
        raw = np.array([row.get(feat, 0.0) for row in over_rows], dtype=np.float32)
        if feat in norm_stats:
            raw = normalise(
                raw, norm_stats[feat]["mean"], norm_stats[feat]["std"]
            ).astype(np.float32)
        step_mat[:, col_idx] = raw

    static_mat = np.tile(static, (n, 1))
    full_mat = np.concatenate([step_mat, static_mat], axis=1)

    x = torch.from_numpy(full_mat).unsqueeze(0).to(device)  # (1, n, 10)
    lengths = torch.tensor([n], dtype=torch.long).to(device)

    model.eval()
    with torch.no_grad():
        pred = model(x, lengths)

    return float(pred.item())
