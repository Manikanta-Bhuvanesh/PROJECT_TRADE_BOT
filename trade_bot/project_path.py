"""Put this repo's project root on ``sys.path`` so ``Algorithms.*`` imports work."""
from __future__ import annotations

import sys
from pathlib import Path


def inject(project_root: Path) -> Path:
    root = project_root.resolve()
    s = str(root)
    if s not in sys.path:
        sys.path.insert(0, s)
    return root
