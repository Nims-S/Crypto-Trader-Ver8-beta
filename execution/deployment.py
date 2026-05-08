from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config.execution import DEFAULT_TOTAL_CAPITAL, LIVE_STATE_FILE
from execution.allocator import allocate_capital
from execution.drift_monitor import compare_performance
from execution.router import route_strategies, select_active_strategy
from registry.store import get_strategy, list_strategies, rank_strategies, upsert_strategy


@dataclass(frozen=True)
class DeploymentPlan:
    mode: str
    created_at: str
    symbols: list[str]
    timeframes: list[str]
    routes: list[dict[str, Any]]
    allocations: list[dict[str, Any]]
    actions: list[dict[str, Any]]
    live_state: dict[str, Any]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _state_path() -> Path:
    return Path(os.getenv("LIVE_STATE_FILE", LIVE_STATE_FILE))


def _load_state() -> dict[str, Any]:
    path = _state_path()
    if not path.exists():
        return {
            "created_at": _now(),
            "updated_at": _now(),
            "deployments": [],
            "live_metrics": {},
            "drift_events": [],
            "allocations": {},
        }
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        data = {}
    if not isinstance(data, dict):
        data = {}
    data.setdefault("created_at", _now())
    data.setdefault("updated_at", _now())
    data.setdefault("deployments", [])
    data.setdefault("live_metrics", {})
    data.setdefault("drift_events", [])
    data.setdefault("allocations", {})
    return data


def _save_state(state: dict[str, Any]) -> None:
    path = _state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _normalize_symbols(symbols: list[str] | tuple[str, ...] | None) -> list[str]:
    if not symbols:
        return []
    return [str(s).strip() for s in symbols if str(s).strip()]


def _normalize_timeframes(timeframes: list[str] | tuple[str, ...] | None) -> list[str]:
    if not timeframes:
        return []
    return [str(t).strip() for t in timeframes if str(t).strip()]


def _pick_routes(
    symbols: list[str],
    timeframes: list[str],
    *,
    regimes: dict[tuple[str, str], str | None] | None,
    limit: int,
) -> list[dict[str, Any]]:
    routed = route_strategies(symbols, timeframes, regimes=regimes or {}, limit=limit)
    if routed:
        return routed

    fallback: list[dict[str, Any]] = []
    for symbol in symbols:
        for timeframe in timeframes:
            row = select_active_strategy(symbol, timeframe, None, limit=limit)
            if row:
                fallback.append(
                    {
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "regime": None,
                        "strategy_id": row.get("strategy_id"),
                        "score": float(((row.get("metrics") or {}).get("walk_forward") or {}).get("score", 0.0) or 0.0),
                        "strategy": row,
                    }
                )
    return fallback


def _cap_context(routes: list[dict[str, Any]], state: dict[str, Any]) -> dict[str, dict[str, Any]]:
    live_metrics = state.get("live_metrics") or {}
    context: dict[str, dict[str, Any]] = {}
    for route in routes:
        sid = str(route.get("strategy_id") or "")
        strategy = route.get("strategy") or {}
        expected = (strategy.get("metrics") or {})
        live = live_metrics.get(sid) or {}
        drift = compare_performance(
            {
                "profit_factor": (expected.get("backtest") or {}).get("profit_factor", expected.get("profit_factor", 0.0)),
                "win_rate": (expected.get("backtest") or {}).get("win_rate", expected.get("win_rate", 0.0)),
                "max_drawdown_pct": (expected.get("backtest") or {}).get("max_drawdown_pct", expected.get("max_drawdown_pct", 0.0)),
            },
            live,
        )
        enabled = drift.get("status") != "disable"
        context[sid] = {
            "enabled": enabled,
            "multiplier": float(drift.get("allocation_multiplier", 1.0) or 1.0),
            "status": drift.get("status", "unknown"),
            "expected": expected,
            "live": live,
            "drift": drift,
        }
    return context


def _flatten_route(route: dict[str, Any]) -> dict[str, Any]:
    strategy = route.get("strategy") or {}
    metrics = strategy.get("metrics") or {}
    flat = dict(strategy)
    flat.setdefault("strategy_id", route.get("strategy_id") or strategy.get("strategy_id"))
    flat.setdefault("symbol", route.get("symbol") or strategy.get("symbol") or (metrics.get("backtest") or {}).get("symbol"))
    flat.setdefault("timeframe", route.get("timeframe") or strategy.get("timeframe") or (metrics.get("backtest") or {}).get("ltf_timeframe"))
    flat.setdefault("regime", route.get("regime") or strategy.get("regime") or strategy.get("regime_profile"))
    flat.setdefault("status", strategy.get("status") or "validated")
    flat.setdefault("metrics", metrics)
    flat.setdefault("tags", strategy.get("tags") or [])
    flat.setdefault("parameters", strategy.get("parameters") or {})
    flat.setdefault("active", strategy.get("active", True))
    return flat


def build_deployment_plan(
    *,
    symbols: list[str] | tuple[str, ...],
    timeframes: list[str] | tuple[str, ...],
    total_capital: float = DEFAULT_TOTAL_CAPITAL,
    regimes: dict[tuple[str, str], str | None] | None = None,
    limit: int = 5,
    temperature: float = 1.0,
    paper: bool = True,
) -> DeploymentPlan:
    symbol_list = _normalize_symbols(symbols)
    timeframe_list = _normalize_timeframes(timeframes)
    if not symbol_list:
        symbol_list = ["BTC/USDT"]
    if not timeframe_list:
        timeframe_list = ["1d"]

    state = _load_state()
    routes = _pick_routes(symbol_list, timeframe_list, regimes=regimes, limit=limit)
    context = _cap_context(routes, state)
    allocation_inputs = [_flatten_route(route) for route in routes]
    allocations = allocate_capital(allocation_inputs, float(total_capital or DEFAULT_TOTAL_CAPITAL), temperature=temperature, context=context)

    actions: list[dict[str, Any]] = []
    for route, alloc in zip(routes, allocations):
        sid = str(route.get("strategy_id") or "")
        ctx = context.get(sid, {})
        actions.append(
            {
                "strategy_id": sid,
                "symbol": route.get("symbol"),
                "timeframe": route.get("timeframe"),
                "regime": route.get("regime"),
                "status": ctx.get("status", "unknown"),
                "enabled": bool(ctx.get("enabled", True)),
                "capital": alloc.get("capital", 0.0),
                "weight": alloc.get("weight", 0.0),
                "score": alloc.get("score", 0.0),
                "mode": "paper" if paper else "live",
            }
        )

    plan = DeploymentPlan(
        mode="paper" if paper else "live",
        created_at=_now(),
        symbols=symbol_list,
        timeframes=timeframe_list,
        routes=routes,
        allocations=allocations,
        actions=actions,
        live_state=state,
    )
    return plan


def commit_deployment_plan(plan: DeploymentPlan) -> dict[str, Any]:
    state = _load_state()
    deployment = {
        "mode": plan.mode,
        "created_at": plan.created_at,
        "symbols": plan.symbols,
        "timeframes": plan.timeframes,
        "routes": plan.routes,
        "allocations": plan.allocations,
        "actions": plan.actions,
    }
    state["deployments"].append(deployment)
    state["updated_at"] = _now()
    state["allocations"] = {a["strategy_id"]: a for a in plan.actions}
    _save_state(state)
    return deployment


def update_live_metric(strategy_id: str, metrics: dict[str, Any]) -> dict[str, Any]:
    state = _load_state()
    live_metrics = state.setdefault("live_metrics", {})
    prev = live_metrics.get(strategy_id, {})
    merged = {**prev, **metrics, "updated_at": _now()}
    live_metrics[strategy_id] = merged
    state["updated_at"] = _now()
    _save_state(state)
    return merged


def evaluate_drift(strategy_id: str) -> dict[str, Any]:
    state = _load_state()
    live = (state.get("live_metrics") or {}).get(strategy_id, {})
    row = get_strategy(strategy_id)
    expected = (row.get("metrics") or {}).get("backtest") or (row.get("metrics") or {}).get("walk_forward") or {}
    drift = compare_performance(expected, live)
    event = {
        "strategy_id": strategy_id,
        "created_at": _now(),
        "drift": drift,
        "live": live,
        "expected": expected,
    }
    state.setdefault("drift_events", []).append(event)
    if drift.get("status") == "disable":
        upsert_strategy(strategy_id, status="disabled", active=False, notes="auto-disabled due to live drift")
    elif drift.get("status") == "throttle":
        upsert_strategy(strategy_id, status="validated", active=True, notes="auto-throttled due to live drift")
    state["updated_at"] = _now()
    _save_state(state)
    return event


def summarize_deployment_state() -> dict[str, Any]:
    state = _load_state()
    deployments = state.get("deployments") or []
    live_metrics = state.get("live_metrics") or {}
    drift_events = state.get("drift_events") or []
    active_allocations = state.get("allocations") or {}
    return {
        "created_at": state.get("created_at"),
        "updated_at": state.get("updated_at"),
        "deployments": len(deployments),
        "live_metric_strategies": len(live_metrics),
        "drift_events": len(drift_events),
        "active_allocations": active_allocations,
        "latest_deployment": deployments[-1] if deployments else None,
        "latest_drift": drift_events[-1] if drift_events else None,
    }
