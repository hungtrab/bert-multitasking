"""Make ``src/`` and the project root importable in tests."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for sub in (ROOT, ROOT / "src"):
    if str(sub) not in sys.path:
        sys.path.insert(0, str(sub))
