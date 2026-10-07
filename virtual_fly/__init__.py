"""Independent brain, body, and translation modules."""
import os
from pathlib import Path

_cache = Path(__file__).resolve().parent.parent / ".cache"
os.environ.setdefault("MPLCONFIGDIR", str(_cache / "matplotlib"))
os.environ.setdefault("FLYGYM_ASSET_CACHE_DIR", str(_cache / "flygym_assets"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(_cache / "numba"))
