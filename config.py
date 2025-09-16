import json
from pathlib import Path

DEFAULT_CONFIG = {
    "tt_size": 2**20,
    "time_limit": 10.0,
    "use_bitboard": True,
    "difficulties": {
        "easy": {"max_depth": 6, "use_parallel": False},
        "medium": {"max_depth": 8, "use_parallel": False},
        "hard": {"max_depth": 10, "use_parallel": False},
    },
}


def load_config(path: str = "config.json") -> dict:
    p = Path(path)
    if not p.exists():
        return DEFAULT_CONFIG.copy()
    try:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
            # Merge with defaults (shallow merge)
            merged = DEFAULT_CONFIG.copy()
            merged.update({k: v for k, v in data.items() if v is not None})
            # Merge nested difficulties if present
            if "difficulties" in data and isinstance(data["difficulties"], dict):
                merged_d = merged.get("difficulties", {}).copy()
                merged_d.update(data["difficulties"])  # override per difficulty
                merged["difficulties"] = merged_d
            return merged
    except Exception:
        return DEFAULT_CONFIG.copy()

