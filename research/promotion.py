from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any

from registry.store import list_strategies, list_experiments, upsert_strategy

ARTIFACT_PATH = Path("artifacts/promotion_report.json")


def _safe(v, d=0.0):
    try:
        return float(v)
    except Exception:
        return d


def _latest_score(row: dict) -> float:
    wf = (row.get("metrics") or {}).get("walk_forward") or {}
    return _safe(wf.get("score", 0.0))


def _has_passed_experiment(strategy_id: str) -> bool:
    exps = list_experiments(strategy_id=strategy_id, limit=25)
    return any(e.get("passed") for e in exps)


def select_candidates(limit: int = 10, min_score: float = 0.55) -> List[Dict[str, Any]]:
    rows = list_strategies(active_only=False)
    eligible = []

    for r in rows:
        score = _latest_score(r)
        if score < min_score:
            continue

        if not _has_passed_experiment(r.get("strategy_id")) and r.get("status") not in {"validated", "architecture_promoted"}:
            continue

        eligible.append((score, r))

    eligible.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in eligible[:limit]]


def promote_winners(limit: int = 10, dry_run: bool = True, min_score: float = 0.55) -> dict:
    winners = select_candidates(limit=limit, min_score=min_score)
    promoted = []

    for row in winners:
        sid = row.get("strategy_id")
        if not dry_run:
            upsert_strategy(
                sid,
                base_strategy=row.get("base_strategy"),
                version=row.get("version"),
                status="architecture_promoted",
                parameters=row.get("parameters"),
                metrics=row.get("metrics"),
                tags=row.get("tags"),
                source="architecture_promotion",
                notes="promoted by system",
                active=True,
                validated_at=datetime.now(timezone.utc).isoformat(),
            )

        promoted.append({
            "strategy_id": sid,
            "score": _latest_score(row),
            "status": row.get("status"),
        })

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "count": len(promoted),
        "strategies": promoted,
    }

    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with ARTIFACT_PATH.open("w") as f:
        json.dump(report, f, indent=2)

    print(f"Promotion report written: {ARTIFACT_PATH}")
    return report
