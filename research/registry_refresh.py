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


def _normalize_entry(entry: dict[str, Any]) -> dict[str, Any]:
    strategy_id = str(entry.get("strategy_id") or entry.get("candidate_id") or "").strip()
    if not strategy_id:
        raise ValueError("registry entry missing strategy_id")

    metrics = entry.get("metrics") or {}
    agent_score = metrics.get("agent_score") or {}
    backtest = metrics.get("backtest") or {}
    walk_forward = metrics.get("walk_forward") or {}
    monte_carlo = metrics.get("monte_carlo") or {}
    perturbation = metrics.get("perturbation") or {}
    cross_symbol = metrics.get("cross_symbol") or {}

    status = str(entry.get("status") or "candidate").strip().lower()
    if status not in STATUS_ORDER:
        status = "candidate"

    return {
        "strategy_id": strategy_id,
        "base_strategy": str(entry.get("base_strategy") or entry.get("parent_strategy_id") or "evolution"),
        "version": int(entry.get("version", 1) or 1),
        "status": status,
        "parameters": entry.get("parameters") or {},
        "metrics": {
            "agent_score": agent_score,
            "backtest": backtest,
            "walk_forward": walk_forward,
            "monte_carlo": monte_carlo,
            "perturbation": perturbation,
            "cross_symbol": cross_symbol,
        },
        "tags": entry.get("tags") or [entry.get("symbol"), entry.get("timeframe"), entry.get("regime")],
        "source": str(entry.get("source") or "snapshot_refresh"),
        "notes": str(entry.get("notes") or ""),
        "active": bool(entry.get("active", False)),
        "validated_at": entry.get("validated_at"),
        "regime_profile": entry.get("regime_profile") or entry.get("regime"),
        "robustness_score": float(entry.get("robustness_score", _robustness_score(metrics)) or 0.0),
        "parent_strategy_id": entry.get("parent_strategy_id"),
        "created_at": entry.get("created_at"),
        "updated_at": entry.get("updated_at"),
        "passed": bool(entry.get("passed", False)),
        "symbol": entry.get("symbol"),
        "timeframe": entry.get("timeframe"),
        "regime": entry.get("regime"),
        "score": float(entry.get("score", 0.0) or 0.0),
    }


def refresh_registry_from_snapshot(snapshot: dict[str, Any], *, source_file: str | None = None) -> dict[str, Any]:
    # Support two formats:
    # 1) Full evolve snapshots with a top-level `results` list.
    # 2) Compact registry manifests with a top-level `registry_entries` list.
    results = snapshot.get("results") or snapshot.get("registry_entries") or []
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

        try:
            if "registry_entries" in snapshot:
                normalized = _normalize_entry(result)
                candidate_id = normalized["strategy_id"]
                metrics = normalized["metrics"]
                parameters = normalized["parameters"]
                symbol = str(normalized.get("symbol") or "")
                timeframe = str(normalized.get("timeframe") or "")
                regime = str(normalized.get("regime") or "unknown") or None
                parent_id = str(normalized.get("parent_strategy_id") or "").strip() or None
                final_status = normalized["status"]
                active = bool(normalized.get("active", False)) or candidate_id in selected_ids or final_status in {"validated", "deployable", "live"}
                robustness_score = float(normalized.get("robustness_score", 0.0) or 0.0)
                passed = bool(normalized.get("passed", False))
                score = float(normalized.get("score", 0.0) or 0.0)
            else:
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
                parameters = result.get("parameters") or {}
                symbol = str(result.get("symbol") or "")
                timeframe = str(result.get("timeframe") or "")
                regime = str(result.get("regime") or "unknown").strip().lower() or None
                parent_id = str(result.get("parent_id") or result.get("parent_strategy_id") or "").strip() or None
                robustness_score = _robustness_score(metrics)
                passed = bool(result.get("passed", False))
                score = float(result.get("score", 0.0) or 0.0)
        except Exception:
            skipped += 1
            continue

        upsert_strategy(
            candidate_id,
            base_strategy=str(result.get("parent_id") or result.get("parent_strategy_id") or normalized.get("base_strategy") if "registry_entries" in snapshot else "evolution"),
            version=int(result.get("version", 1) or 1),
            status=final_status,
            parameters=parameters,
            metrics=metrics,
            tags=[
                str(symbol or result.get("symbol") or "").strip(),
                str(timeframe or result.get("timeframe") or "").strip(),
                str(regime or result.get("regime") or "").strip(),
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
            symbol=str(symbol or result.get("symbol") or ""),
            timeframe=str(timeframe or result.get("timeframe") or ""),
            run_type="snapshot_refresh",
            parameters=parameters,
            metrics=metrics,
            passed=passed,
            notes=f"source={source_file or 'snapshot'}",
        )
        record_evolution_run(
            cycle_id=str(snapshot.get("cycle_id") or "snapshot_refresh"),
            symbol=str(symbol or result.get("symbol") or ""),
            timeframe=str(timeframe or result.get("timeframe") or ""),
            parent_strategy_id=parent_id,
            child_strategy_id=candidate_id,
            status=final_status,
            score=score,
            passed=passed,
            parameters=parameters,
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
