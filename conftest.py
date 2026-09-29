"""Make the repository root importable so tests can ``import core``.

Without this, pytest puts ``tests/`` on the path but not the project root, and
``core`` is unimportable unless the suite is run from the repository root.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT: Path = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
