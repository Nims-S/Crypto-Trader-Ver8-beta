from __future__ import annotations

import uuid
from types import SimpleNamespace
from typing import Any, Sequence

from registry.store import list_strategies, rank_strategies
from research.loop import _persist_evaluation, evaluate_candidate


_DEFAULT_SYMBOLS = ("BTC/USDT", "ETH/USDT", "SOL/USDT")
_DEFAULT_TIMEFRAMES = ("1d",)


def _safe_str(value: Any, default: str = "") -> str:
    try:
        text = str(value or "").strip()
        return text or default
    except Exception:
        return default


def _infer_symbol_timeframe(row: dict[str, Any]) -> tuple[str, str]:
    metrics = row.get("metrics") or {}
    bt = metrics.get("backtest") or {}
    symbol = _safe_str(bt.get("symbol") or row.get("symbol") or row.get("market"))
    timeframe = _safe_str(bt.get("ltf_timeframe") or row.get("timeframe") or bt.get("timeframe"))

    tags = [str(t).strip().lower() for t in (row.get("tags") or []) if str(t).strip()]
    if not symbol:
        for tag in tags:
            if "/" in tag and tag.endswith("usdt"):
                symbol = tag.upper()
                break
    if not timeframe:
        for tag in tags:
            if tag in {"1d", "4h", "1h", "1w", "15m", "30m", "5m"}:
                timeframe = tag
                break
    return symbol, timeframe


def _row_needs_refresh(row: dict[str, Any]) -> bool:
    metrics = row.get("metrics") or {}
    mc = metrics.get("monte_carlo") or {}
    perturb = metrics.get("perturbation") or {}
    cross_symbol = metrics.get("cross_symbol") or {}
    status = _safe_str(row.get("status"), "candidate")

    return (
        not bool(mc.get("passed", False))
        or not bool(perturb.get("passed", False))
        or not bool(cross_symbol.get("passed", False))
        or status in {"candidate", "validated", "deployable"}
    )


def _matches_filters(row: dict[str, Any], symbols: Sequence[str] | None, timeframes: Sequence[str] | None) -> bool:
    symbol, timeframe = _infer_symbol_timeframe(row)
    if symbols:
        symbol_set = {str(s).strip().lower() for s in symbols if str(s).strip()}
        if symbol and symbol.lower() not in symbol_set:
            return False
    if timeframes:
        timeframe_set = {str(t).strip().lower() for t in timeframes if str(t).strip()}
        if timeframe and timeframe.lower() not in timeframe_set:
            return False
    return True


def backfill_registry_evidence(
    *,
    limit: int = 8,
    start: str = "2022-01-01",
    end: str = "2026-04-30",
    folds: int = 4,
    mc_iterations: int = 1000,
    allow_shorts: bool = False,
    use_cache: bool = True,
    validation_symbols: Sequence[str] | None = ("BTC/USDT", "ETH/USDT", "SOL/USDT"),
    symbols: Sequence[str] | None = None,
    timeframes: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Refresh persisted evidence for the current top strategies."""

    limit = max(1, int(limit or 1))
    cycle_id = f"backfill_{uuid.uuid4().hex[:8]}"
    candidates = rank_strategies(active_only=False, limit=max(limit * 3, limit))

    refreshed: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for row in candidates:
        if len(refreshed) >= limit:
            break
        if not _matches_filters(row, symbols=symbols, timeframes=timeframes):
            continue
        if not _row_needs_refresh(row):
            continue

        symbol, timeframe = _infer_symbol_timeframe(row)
        if not symbol or not timeframe:
            skipped.append({"strategy_id": row.get("strategy_id"), "reason": "missing_symbol_timeframe"})
            continue

        params = dict(row.get("parameters") or {})
        candidate = SimpleNamespace(
            strategy_id=row.get("strategy_id"),
            base_strategy=row.get("base_strategy") or row.get("strategy_id") or "seed",
            version=int(row.get("version", 1) or 1),
            parameters=params,
            tags=list(row.get("tags") or []),
        )

        report = evaluate_candidate(
            candidate=candidate,
            parent=row,
            symbol=symbol,
            timeframe=timeframe,
            start=start,
            end=end,
            folds=folds,
            allow_shorts=allow_shorts,
            use_cache=use_cache,
            mc_iterations=mc_iterations,
            validation_symbols=tuple(validation_symbols or (_DEFAULT_SYMBOLS)),
        )

        if "error" in report:
            skipped.append({"strategy_id": row.get("strategy_id"), "reason": report["error"]})
            continue

        _persist_evaluation(
            candidate=candidate,
            parent=row,
            report=report,
            symbol=symbol,
            timeframe=timeframe,
            cycle_id=cycle_id,
        )
        refreshed.append(
            {
                "strategy_id": row.get("strategy_id"),
                "symbol": symbol,
                "timeframe": timeframe,
                "status": report.get("status"),
                "passed": bool(report.get("passed", False)),
                "regime": report.get("regime"),
                "score": float(report.get("score", 0.0) or 0.0),
            }
        )

    return {
        "cycle_id": cycle_id,
        "refreshed_count": len(refreshed),
        "skipped_count": len(skipped),
        "refreshed": refreshed,
        "skipped": skipped,
        "window": {"start": start, "end": end, "folds": folds, "mc_iterations": mc_iterations},
        "symbols": list(symbols or _DEFAULT_SYMBOLS),
        "timeframes": list(timeframes or _DEFAULT_TIMEFRAMES),
    }
