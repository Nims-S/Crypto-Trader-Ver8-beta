from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from registry.store import (
    STATUS_ORDER,
    classify_strategy_status,
    record_evolution_run,
    record_experiment,
    upsert_strategy,
)


def _safe_get(mapping: dict[str, Any] | None, *keys: str, default: Any = None) -> Any:
    cur: Any = mapping or {}
    for key in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
    return default if cur is None else cur


def _load_snapshot(snapshot_file: str | Path) -> dict[str, Any]:
    path = Path(snapshot_file)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("snapshot root must be a JSON object")
    return data


def _status_priority(value: str | None) -> int:
    return STATUS_ORDER.get(str(value or "candidate").strip().lower(), STATUS_ORDER["candidate"])


def _robustness_score(metrics: dict[str, Any]) -> float:
    for key in ("monte_carlo", "perturbation", "walk_forward"):
        block = metrics.get(key) or {}
        score = block.get("score")
        if score is not None:
            try:
                return float(score)
            except Exception:
                continue
    return 0.0


def refresh_registry_from_snapshot(snapshot: dict[str, Any], *, source_file: str | None = None) -> dict[str, Any]:
    results = snapshot.get("results") or []
    portfolio_summary = snapshot.get("portfolio_summary") or {}
    selected_ids = {
        str(row.get("strategy_id") or "").strip()
        for row in (portfolio_summary.get("selected") or [])
        if isinstance(row, dict) and str(row.get("strategy_id") or "").strip()
    }

    refreshed = 0
    skipped = 0
    imported_ids: list[str] = []

    for result in results:
        if not isinstance(result, dict):
            skipped += 1
            continue

        candidate_id = str(result.get("candidate_id") or result.get("strategy_id") or "").strip()
        if not candidate_id:
            skipped += 1
            continue

        metrics = result.get("metrics") or {}
        agent_score = metrics.get("agent_score") or {}
        backtest = metrics.get("backtest") or {}
        walk_forward = metrics.get("walk_forward") or {}

        classification = classify_strategy_status(
            agent_score=agent_score,
            backtest=backtest,
            walk_forward=walk_forward,
            timeframe=result.get("timeframe"),
        )
        snapshot_status = str(result.get("status") or classification.get("status") or "candidate").strip().lower()
        classification_status = str(classification.get("status") or "candidate").strip().lower()
        final_status = max((snapshot_status, classification_status), key=_status_priority)
        active = bool(result.get("passed", False)) or candidate_id in selected_ids or final_status in {"validated", "deployable", "live"}

        robustness_score = _robustness_score(metrics)
        regime = str(result.get("regime") or "unknown").strip().lower() or None
        parent_id = str(result.get("parent_id") or result.get("parent_strategy_id") or "").strip() or None

        upsert_strategy(
            candidate_id,
            base_strategy=str(result.get("parent_id") or result.get("parent_strategy_id") or "evolution"),
            version=1,
            status=final_status,
            parameters=result.get("parameters") or {},
            metrics=metrics,
            tags=[
                str(result.get("symbol") or "").strip(),
                str(result.get("timeframe") or "").strip(),
                str(result.get("regime") or "").strip(),
                "snapshot_refresh",
            ],
            source="snapshot_refresh",
            notes=f"source={source_file or 'snapshot'}; cycle={snapshot.get('cycle_id', 'unknown')}",
            active=active,
            validated_at=result.get("validated_at"),
            regime_profile=regime,
            robustness_score=robustness_score,
            parent_strategy_id=parent_id,
        )
        record_experiment(
            candidate_id,
            symbol=str(result.get("symbol") or ""),
            timeframe=str(result.get("timeframe") or ""),
            run_type="snapshot_refresh",
            parameters=result.get("parameters") or {},
            metrics=metrics,
            passed=bool(result.get("passed", False)),
            notes=f"source={source_file or 'snapshot'}",
        )
        record_evolution_run(
            cycle_id=str(snapshot.get("cycle_id") or "snapshot_refresh"),
            symbol=str(result.get("symbol") or ""),
            timeframe=str(result.get("timeframe") or ""),
            parent_strategy_id=parent_id,
            child_strategy_id=candidate_id,
            status=final_status,
            score=float(result.get("score", 0.0) or 0.0),
            passed=bool(result.get("passed", False)),
            parameters=result.get("parameters") or {},
            metrics=metrics,
            notes=f"source={source_file or 'snapshot'}; selected={candidate_id in selected_ids}",
        )
        refreshed += 1
        imported_ids.append(candidate_id)

    return {
        "cycle_id": snapshot.get("cycle_id"),
        "source_file": source_file,
        "results_seen": len(results),
        "refreshed": refreshed,
        "skipped": skipped,
        "selected_ids": sorted(selected_ids),
        "imported_strategy_ids": imported_ids[:25],
        "portfolio_summary": portfolio_summary,
    }


def refresh_registry_from_snapshot_file(snapshot_file: str | Path) -> dict[str, Any]:
    return refresh_registry_from_snapshot(_load_snapshot(snapshot_file), source_file=str(snapshot_file))
