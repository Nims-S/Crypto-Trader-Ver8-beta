from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from registry.store import rank_strategies


@dataclass(frozen=True)
class RoutedStrategy:
    strategy_id: str
    symbol: str | None
    timeframe: str | None
    regime: str | None
    score: float
    strategy: dict[str, Any]


def select_active_strategy(
    symbol: str,
    timeframe: str,
    regime: str | None = None,
    *,
    limit: int = 5,
) -> dict[str, Any] | None:
    candidates = rank_strategies(symbol=symbol, timeframe=timeframe, regime=regime, active_only=True, limit=max(1, int(limit)))
    if candidates:
        return candidates[0]

    fallback = rank_strategies(symbol=symbol, timeframe=timeframe, regime=None, active_only=True, limit=max(1, int(limit)))
    if fallback:
        return fallback[0]

    fallback = rank_strategies(symbol=symbol, timeframe=timeframe, regime=None, active_only=False, limit=max(1, int(limit)))
    return fallback[0] if fallback else None


def route_strategies(
    symbols: list[str],
    timeframes: list[str],
    regimes: dict[tuple[str, str], str | None] | None = None,
    *,
    limit: int = 5,
) -> list[dict[str, Any]]:
    routed: list[dict[str, Any]] = []
    regimes = regimes or {}
    for symbol in symbols:
        for timeframe in timeframes:
            regime = regimes.get((symbol, timeframe))
            row = select_active_strategy(symbol, timeframe, regime, limit=limit)
            if not row:
                continue

            score = float(((row.get("metrics") or {}).get("walk_forward") or {}).get("score", 0.0) or 0.0)
            routed.append(
                {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "regime": regime,
                    "strategy_id": row.get("strategy_id"),
                    "score": score,
                    "strategy": row,
                }
            )
    return routed
