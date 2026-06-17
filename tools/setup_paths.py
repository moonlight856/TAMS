"""
Put TAMS repo root on sys.path so `python tools/train.py` works from project root
without installing the package or setting PYTHONPATH (Windows / Linux / macOS).
"""
import os
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_root_str = str(_ROOT)
if _root_str not in sys.path:
    sys.path.insert(0, _root_str)
