#!/usr/bin/env python3
"""Make the vendored TIRO submodule importable.

TIRO (Arditi & Varri 2025) ships as loose modules -- TIRO.py, Critical.py,
NBody.py, Physical_Units.py -- at the root of its own repository rather than as
an installable package, so consuming it as a git submodule means putting that
directory on sys.path.  Import this module before importing from TIRO:

    import tiro_path  # noqa: F401
    from TIRO import Model
"""

import sys
from pathlib import Path

TIRO_DIR = Path(__file__).resolve().parent / "TIRO"

if not (TIRO_DIR / "TIRO.py").exists():
    raise ImportError(
        f"TIRO submodule not found at {TIRO_DIR}. Populate it with:\n"
        "    git submodule update --init --recursive"
    )

if str(TIRO_DIR) not in sys.path:
    sys.path.insert(0, str(TIRO_DIR))
