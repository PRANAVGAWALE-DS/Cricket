"""
tests/conftest.py
-----------------
pytest configuration: ensures src/ and api/ are on sys.path so all
test imports resolve correctly regardless of where pytest is invoked.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
