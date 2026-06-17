"""Model factory: delegates to tools/model.build_model for TAMS construction."""

from __future__ import annotations

import sys
from pathlib import Path


def build_model(cfg, device=None):
    """
    Dispatch to ``tools/model.build_model`` for model construction.
    """
    root = Path(__file__).resolve().parents[2]
    tools_dir = str(root / "tools")
    if tools_dir not in sys.path:
        sys.path.insert(0, tools_dir)
    import model as m

    model = m.build_model(cfg)
    if device is not None:
        model = model.to(device)
    return model
