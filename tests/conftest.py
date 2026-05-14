from __future__ import annotations

import sys
from pathlib import Path

# tests/conftest.py is a pytest special file:
# pytest imports it automatically before running tests.

# Figure out the absolute path to the project's src/ directory.
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    # Add src/ to Python import path so tests can import the project package without installation.
    sys.path.insert(0, str(SRC))
