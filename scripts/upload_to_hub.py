"""
scripts/upload_to_hub.py
------------------------
One-time script: uploads all model artefacts and processed parquets
to the HF Hub dataset repo that the Space downloads at startup.

Run once from the project root (after training all models):
    pip install huggingface_hub
    huggingface-cli login          # enter a write-scoped HF token
    # or set HF_TOKEN / HUGGING_FACE_HUB_TOKEN in your shell
    python scripts/upload_to_hub.py

HF Hub repo created: PG-AIML/cricket-ml-artefacts (private by default)
The Space (start.sh) downloads from this repo at container startup.

Files uploaded
--------------
models/
    match_winner.ubj
    potm_classifier.ubj
    score_predictor.pkl
    win_probability.pkl
    gru_score_predictor.pt       (if present)

data/processed/
    deliveries.parquet
    match_features_v3.parquet
    match_features.parquet
    matches.parquet
    potm_features.parquet
    score_features.parquet
    team_rolling_form.parquet
    win_prob_features.parquet
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from huggingface_hub import HfApi, create_repo, get_token
    from huggingface_hub.errors import HfHubHTTPError
except ImportError:
    print("ERROR: huggingface_hub not installed.")
    print("Run: pip install huggingface_hub")
    sys.exit(1)

REPO_ID = "PG-AIML/cricket-ml-artefacts"
REPO_TYPE = "dataset"

# Model extensions to upload
MODEL_EXTS = {".ubj", ".pkl", ".pt", ".pth"}


def die_auth(message: str) -> None:
    print()
    print("ERROR: Hugging Face authentication failed.")
    print(message)
    print()
    print("Fix:")
    print("  1. Create a Hugging Face access token with write permission:")
    print("     https://huggingface.co/settings/tokens")
    print("  2. Log in again from this same virtualenv/session:")
    print("     huggingface-cli logout")
    print("     huggingface-cli login")
    print("  3. Make sure the token's user can create/write to the PG-AIML org.")
    print()
    print("Alternative for this PowerShell session:")
    print("  $env:HF_TOKEN = \"hf_your_token_here\"")
    print("  python scripts/upload_to_hub.py")
    sys.exit(1)


def main() -> None:
    token = get_token()
    if not token:
        die_auth("No HF token was found in the login cache or environment.")

    api = HfApi(token=token)
    try:
        user = api.whoami(token=token)
    except HfHubHTTPError as exc:
        die_auth(f"The configured HF token was rejected: {exc}")

    username = user.get("name") or user.get("fullname") or "authenticated user"
    print(f"Authenticated to HF Hub as: {username}")

    # Create repo if it doesn't exist (private by default)
    print(f"Creating/verifying HF Hub repo: {REPO_ID}")
    try:
        create_repo(
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            private=True,  # set False to make public
            exist_ok=True,
            token=token,
        )
    except HfHubHTTPError as exc:
        die_auth(
            "Could not create or verify the repo. "
            f"Check that your token has write permission for {REPO_ID}. "
            f"Original error: {exc}"
        )
    print("  Repo ready.")
    print()

    total_bytes = 0
    uploaded = 0

    # ── Upload models ──────────────────────────────────────────────────────
    models_dir = ROOT / "models"
    if not models_dir.exists():
        print(f"ERROR: {models_dir} not found. Run training notebooks first.")
        sys.exit(1)

    print("Uploading models/...")
    for f in sorted(models_dir.iterdir()):
        if f.suffix.lower() not in MODEL_EXTS:
            continue
        size_kb = f.stat().st_size // 1024
        print(f"  {f.name:45s} {size_kb:>6} KB", end="  ", flush=True)
        api.upload_file(
            path_or_fileobj=str(f),
            path_in_repo=f"models/{f.name}",
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            commit_message=f"Upload model: {f.name}",
        )
        total_bytes += f.stat().st_size
        uploaded += 1
        print("✅")

    # ── Upload processed parquets ──────────────────────────────────────────
    proc_dir = ROOT / "data" / "processed"
    if not proc_dir.exists():
        print(f"ERROR: {proc_dir} not found. Run feature engineering notebooks first.")
        sys.exit(1)

    print()
    print("Uploading data/processed/...")
    for f in sorted(proc_dir.glob("*.parquet")):
        size_kb = f.stat().st_size // 1024
        print(f"  {f.name:45s} {size_kb:>6} KB", end="  ", flush=True)
        api.upload_file(
            path_or_fileobj=str(f),
            path_in_repo=f"data/processed/{f.name}",
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            commit_message=f"Upload parquet: {f.name}",
        )
        total_bytes += f.stat().st_size
        uploaded += 1
        print("✅")

    # ── Summary ────────────────────────────────────────────────────────────
    print()
    print("=" * 55)
    print(f"Uploaded {uploaded} files ({total_bytes / 1024 / 1024:.1f} MB total)")
    print(f"Repo: https://huggingface.co/datasets/{REPO_ID}")
    print()
    print("Next steps:")
    print("  1. Push code to HF Space:")
    print("     git remote add space https://huggingface.co/spaces/PG-AIML/Cricket")
    print("     git push space main")
    print("  2. Watch build logs at:")
    print("     https://huggingface.co/spaces/PG-AIML/Cricket")
    print("=" * 55)


if __name__ == "__main__":
    main()
