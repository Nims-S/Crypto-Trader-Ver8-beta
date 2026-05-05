"""Persistence helpers for strategy evolution (extended, backward compatible)."""

from __future__ import annotations

import hashlib
import json
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List

_STORE_PATH = Path(os.getenv("STRATEGY_STORE_FILE", ".strategy_store.json"))
_STORE_LOCK = threading.RLock()

STATUS_ORDER = {
    "retired": 0,
    "disabled": 1,
    "candidate": 2,
    "validated": 3,
    "deployable": 4,
    "live": 5,
}

VALIDATED_SCORE_THRESHOLD = 0.70
VALIDATED_DENSITY_FLOOR = 0.35
DEPLOYABLE_DENSITY_FLOOR = 0.60
DEPLOYABLE_MAX_DRAWDOWN_PCT = 15.0
DEPLOYABLE_MIN_TRADES = 12


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_status(status: Any) -> str:
    value = str(status or "candidate").strip().lower()
    return value if value in STATUS_ORDER else "candidate"


def _status_rank(status: Any) -> int:
    return STATUS_ORDER.get(_normalize_status(status), STATUS_ORDER["candidate"])


def _composite_score(r: dict[str, Any]) -> float:
    m = r.get("metrics") or {}
    agent = m.get("agent_score") or {}
    wf = m.get("walk_forward") or {}
    bt = m.get("backtest") or {}

    agent_score = float(agent.get("score", 0.0) or 0.0)
    wf_score = float(wf.get("score", 0.0) or 0.0)
    bt_return = max(0.0, float(bt.get("return_pct", 0.0) or 0.0))
    robustness = float(r.get("robustness_score", 0.0) or 0.0)

    return (
        0.45 * agent_score
        + 0.25 * wf_score
        + 0.15 * min(bt_return / 2.0, 1.0)
        + 0.15 * robustness
    )


def compute_logic_hash(parameters: dict[str, Any] | None) -> str:
    try:
        blob = json.dumps(parameters or {}, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]
    except Exception:
        return "unknown"


def _load() -> dict[str, Any]:
    if not _STORE_PATH.exists():
        return {"registry": {}, "experiments": [], "evolution_runs": [], "counters": {"experiment_id": 0, "evolution_id": 0}}
    try:
        with _STORE_PATH.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        data = {}
    data.setdefault("registry", {})
    data.setdefault("experiments", [])
    data.setdefault("evolution_runs", [])
    data.setdefault("counters", {"experiment_id": 0, "evolution_id": 0})
    return data


def _save(store: dict[str, Any]) -> None:
    with _STORE_LOCK:
        tmp = _STORE_PATH.with_name(f"{_STORE_PATH.stem}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(store, fh, indent=2, sort_keys=True, default=str)
        os.replace(tmp, _STORE_PATH)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def _row(strategy_id: str, row: dict[str, Any] | None) -> dict[str, Any]:
    if not row:
        return {}
    return {
        "strategy_id": strategy_id,
        "base_strategy": row.get("base_strategy", "unknown"),
        "version": int(row.get("version", 1) or 1),
        "status": _normalize_status(row.get("status", "candidate")),
        "parameters": row.get("parameters", {}) or {},
        "metrics": row.get("metrics", {}) or {},
        "tags": row.get("tags", []) or [],
        "source": row.get("source", "manual"),
        "notes": row.get("notes", "") or "",
        "active": bool(row.get("active", False)),
        "logic_hash": row.get("logic_hash"),
        "regime_profile": row.get("regime_profile"),
        "robustness_score": float(row.get("robustness_score", 0.0) or 0.0),
        "parent_strategy_id": row.get("parent_strategy_id"),
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
        "validated_at": row.get("validated_at"),
    }


def classify_strategy_status(
    *,
    agent_score: dict[str, Any] | None = None,
    backtest: dict[str, Any] | None = None,
    walk_forward: dict[str, Any] | None = None,
    timeframe: str | None = None,
) -> dict[str, Any]:
    agent_score = agent_score or {}
    backtest = backtest or {}
    walk_forward = walk_forward or {}

    score = float(agent_score.get("score", 0.0) or 0.0)
    score_passed = bool(agent_score.get("passed", False))
    score_reasons = [str(r) for r in (agent_score.get("reasons") or [])]

    bt_return = float(backtest.get("return_pct", 0.0) or 0.0)
    bt_pf = float(backtest.get("profit_factor", 0.0) or 0.0)
    bt_wr = float(backtest.get("win_rate", 0.0) or 0.0)
    bt_dd = abs(float(backtest.get("max_drawdown_pct", 0.0) or 0.0))
    bt_trades = int(backtest.get("trades", 0) or 0)

    wf_passed = bool(walk_forward.get("passed", False))
    wf_score = float(walk_forward.get("score", 0.0) or 0.0)
    wf_spread = abs(float(walk_forward.get("score_spread", 0.0) or 0.0))
    wf_density = float(walk_forward.get("density_mean", 0.0) or 0.0)
    wf_reasons = [str(r) for r in (walk_forward.get("reasons") or [])]

    deployment_quality = (
        score_passed
        and wf_passed
        and bt_trades >= DEPLOYABLE_MIN_TRADES
        and wf_density >= DEPLOYABLE_DENSITY_FLOOR
        and bt_dd <= DEPLOYABLE_MAX_DRAWDOWN_PCT
        and bt_return >= 0.0
        and bt_pf >= 0.95
        and bt_wr >= 0.40
        and wf_spread <= 0.40
    )

    validation_quality = (
        score >= VALIDATED_SCORE_THRESHOLD
        and bt_return >= 0.0
        and bt_pf >= 0.95
        and bt_wr >= 0.40
        and bt_dd <= DEPLOYABLE_MAX_DRAWDOWN_PCT
        and (wf_score >= VALIDATED_SCORE_THRESHOLD * 0.55 or wf_density >= VALIDATED_DENSITY_FLOOR)
    )

    if deployment_quality:
        status = "deployable"
    elif validation_quality:
        status = "validated"
    else:
        status = "candidate"

    active = status in {"deployable", "live"}
    reasons = list(dict.fromkeys(score_reasons + wf_reasons))
    return {
        "status": status,
        "active": active,
        "score": score,
        "score_passed": score_passed,
        "walk_forward_passed": wf_passed,
        "reasons": reasons,
    }


def upsert_strategy(
    strategy_id: str,
    *,
    base_strategy: str = "unknown",
    version: int = 1,
    status: str = "candidate",
    parameters: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
    tags: list[str] | None = None,
    source: str = "manual",
    notes: str = "",
    active: bool = False,
    validated_at: datetime | str | None = None,
    regime_profile: str | None = None,
    robustness_score: float = 0.0,
    parent_strategy_id: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    with _STORE_LOCK:
        store = _load()
        now = _now()
        existing = store["registry"].get(strategy_id, {}) or {}

        extras = dict(kwargs or {})
        if extras:
            params = dict(parameters or {})
            meta = dict(params.get("_meta") or {})
            meta.update({str(k): _jsonable(v) for k, v in extras.items()})
            params["_meta"] = meta
            parameters = params

        logic_hash = compute_logic_hash(parameters)
        incoming_status = _normalize_status(status)
        existing_status = _normalize_status(existing.get("status", "candidate"))
        final_status = incoming_status if _status_rank(incoming_status) >= _status_rank(existing_status) else existing_status
        final_active = bool(active) or bool(existing.get("active", False)) or final_status in {"deployable", "live"}

        row = {
            "base_strategy": base_strategy,
            "version": int(version or 1),
            "status": final_status,
            "parameters": _jsonable(parameters or {}),
            "metrics": _jsonable(metrics or {}),
            "tags": _jsonable(tags or []),
            "source": source,
            "notes": notes,
            "active": bool(final_active),
            "logic_hash": logic_hash,
            "regime_profile": regime_profile,
            "robustness_score": float(robustness_score or 0.0),
            "parent_strategy_id": parent_strategy_id,
            "created_at": existing.get("created_at", now),
            "updated_at": now,
            "validated_at": validated_at.isoformat() if hasattr(validated_at, "isoformat") else validated_at,
        }
        store["registry"][strategy_id] = row
        _save(store)
        return _row(strategy_id, row)


def record_experiment(
    strategy_id: str,
    *,
    symbol: str,
    timeframe: str,
    run_type: str = "backtest",
    parameters: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
    passed: bool = False,
    notes: str = "",
) -> dict[str, Any]:
    with _STORE_LOCK:
        store = _load()
        store["counters"]["experiment_id"] = int(store["counters"].get("experiment_id", 0)) + 1
        row = {
            "id": store["counters"]["experiment_id"],
            "strategy_id": strategy_id,
            "symbol": symbol,
            "timeframe": timeframe,
            "run_type": run_type,
            "parameters": _jsonable(parameters or {}),
            "metrics": _jsonable(metrics or {}),
            "passed": bool(passed),
            "notes": notes,
            "created_at": _now(),
        }
        store["experiments"].append(row)
        _save(store)
        return row


def record_evolution_run(
    *,
    cycle_id: str,
    symbol: str,
    timeframe: str,
    parent_strategy_id: str | None,
    child_strategy_id: str,
    status: str,
    score: float = 0.0,
    passed: bool = False,
    parameters: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
    notes: str = "",
) -> dict[str, Any]:
    with _STORE_LOCK:
        store = _load()
        store["counters"]["evolution_id"] = int(store["counters"].get("evolution_id", 0)) + 1
        row = {
            "id": store["counters"]["evolution_id"],
            "cycle_id": cycle_id,
            "symbol": symbol,
            "timeframe": timeframe,
            "parent_strategy_id": parent_strategy_id,
            "child_strategy_id": child_strategy_id,
            "status": _normalize_status(status),
            "score": float(score),
            "passed": bool(passed),
            "parameters": _jsonable(parameters or {}),
            "metrics": _jsonable(metrics or {}),
            "notes": notes,
            "created_at": _now(),
        }
        store["evolution_runs"].append(row)
        _save(store)
        return row


def list_strategies(active_only: bool = False) -> list[dict[str, Any]]:
    store = _load()
    rows = [_row(strategy_id, row) for strategy_id, row in store["registry"].items()]
    if active_only:
        rows = [row for row in rows if row.get("active")]
    rows.sort(key=lambda r: (r.get("updated_at") or "", r.get("created_at") or ""), reverse=True)
    return rows


def rank_strategies(
    *,
    symbol: str | None = None,
    timeframe: str | None = None,
    regime: str | None = None,
    active_only: bool = True,
    limit: int = 10,
) -> List[dict[str, Any]]:
    rows = list_strategies(active_only=active_only)

    def _match(r):
        tags = {str(t).lower() for t in (r.get("tags") or [])}
        if symbol and symbol.lower() not in tags:
            return False
        if timeframe and timeframe.lower() not in tags:
            return False
        if regime and (r.get("regime_profile") or "") != regime:
            return False
        return True

    rows = [r for r in rows if _match(r)]

    rows.sort(
        key=lambda r: (
            _composite_score(r),
            _status_rank(r.get("status")),
            float(r.get("robustness_score", 0.0)),
            r.get("updated_at") or "",
        ),
        reverse=True,
    )
    return rows[:limit]


def get_strategy(strategy_id: str) -> dict[str, Any]:
    store = _load()
    return _row(strategy_id, store["registry"].get(strategy_id))


def list_experiments(strategy_id: str | None = None, limit: int = 100, run_type: str | None = None) -> list[dict[str, Any]]:
    store = _load()
    rows = [e for e in store["experiments"] if strategy_id is None or e.get("strategy_id") == strategy_id]
    if run_type is not None:
        rows = [e for e in rows if str(e.get("run_type") or "") == str(run_type)]
    rows.sort(key=lambda r: r.get("created_at") or "", reverse=True)
    return rows[: max(1, int(limit))]


def list_evolution_runs(strategy_id: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
    store = _load()
    rows = [r for r in store["evolution_runs"] if strategy_id is None or r.get("child_strategy_id") == strategy_id]
    rows.sort(key=lambda r: r.get("created_at") or "", reverse=True)
    return rows[: max(1, int(limit))]


def export_trade_history(strategy_id: str | None = None, limit: int = 1000, run_type: str | None = None) -> list[dict[str, Any]]:
    rows = list_experiments(strategy_id=strategy_id, limit=max(1, int(limit)), run_type=run_type)
    out: list[dict[str, Any]] = []
    for row in rows:
        metrics = row.get("metrics") or {}
        detail = metrics.get("trades_detail")
        if isinstance(detail, list) and detail:
            for t in detail:
                if not isinstance(t, dict):
                    continue
                out.append(
                    {
                        "strategy_id": row.get("strategy_id"),
                        "symbol": row.get("symbol"),
                        "timeframe": row.get("timeframe"),
                        "run_type": row.get("run_type"),
                        "created_at": row.get("created_at"),
                        "passed": bool(row.get("passed")),
                        "pnl": float(t.get("pnl", 0.0) or 0.0),
                        "entry_price": t.get("entry_price"),
                        "exit_price": t.get("exit_price"),
                        "qty": t.get("qty"),
                        "reason": t.get("reason"),
                        "source": "registry_experiment",
                        "raw": t,
                    }
                )
            continue

        trade = metrics.get("trade") or metrics.get("close_result") or metrics.get("execution") or {}
        if not isinstance(trade, dict):
            trade = {}
        out.append(
            {
                "strategy_id": row.get("strategy_id"),
                "symbol": row.get("symbol"),
                "timeframe": row.get("timeframe"),
                "run_type": row.get("run_type"),
                "created_at": row.get("created_at"),
                "passed": bool(row.get("passed")),
                "pnl": float(trade.get("pnl", 0.0) or 0.0),
                "entry_price": trade.get("entry_price"),
                "exit_price": trade.get("exit_price"),
                "qty": trade.get("qty"),
                "reason": trade.get("reason"),
                "source": "registry_experiment",
                "raw": row,
            }
        )
    return out
