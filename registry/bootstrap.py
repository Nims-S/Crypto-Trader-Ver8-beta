"""Bootstrap helpers for local strategy registry state."""

from __future__ import annotations

from pathlib import Path
import json
import os


def _store_path() -> Path:
    return Path(os.getenv("STRATEGY_STORE_FILE", ".strategy_store.json"))


def init_db() -> dict:
    """Ensure the strategy store exists and return a light bootstrap payload."""
    store_path = _store_path()
    if not store_path.exists():
        payload = {
            "counters": {"experiment_id": 0, "evolution_id": 0},
            "registry": {},
            "experiments": [],
            "evolution_runs": [],
        }
        store_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return {"initialized": True, "store_path": str(store_path)}

    # Validate JSON shape without mutating existing data.
    try:
        data = json.loads(store_path.read_text(encoding="utf-8"))
    except Exception:
        # Preserve the broken file for inspection; let callers decide how to react.
        return {"initialized": False, "store_path": str(store_path), "status": "invalid_json"}

    data.setdefault("counters", {}).setdefault("experiment_id", 0)
    data.setdefault("counters", {}).setdefault("evolution_id", 0)
    data.setdefault("registry", {})
    data.setdefault("experiments", [])
    data.setdefault("evolution_runs", [])
    return {"initialized": True, "store_path": str(store_path), "status": "ok"}
