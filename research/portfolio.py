from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from math import sqrt
from statistics import mean
from typing import Any, Iterable, Sequence

from research.monte_carlo import infer_regime_hint


@dataclass(frozen=True)
class PortfolioCandidate:
    strategy_id: str
    symbol: str
    timeframe: str
    regime: str
    score: float
    row: dict[str, Any]


@dataclass(frozen=True)
class PortfolioEvaluation:
    selected: list[dict[str, Any]]
    weights: list[float]
    summary: dict[str, Any]
    curve: list[dict[str, Any]]


_BASE_EQUITY = 10_000.0


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _parse_time(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    text = str(value or "").strip()
    if not text:
        return datetime.min
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception:
        return datetime.min


def _market_pair(row: dict[str, Any]) -> tuple[str, str]:
    metrics = row.get("metrics") or {}
    bt = metrics.get("backtest") or {}
    symbol = str(bt.get("symbol") or row.get("symbol") or row.get("market") or "unknown")
    timeframe = str(bt.get("ltf_timeframe") or row.get("timeframe") or bt.get("timeframe") or "unknown")
    return symbol, timeframe


def _regime_from_row(row: dict[str, Any]) -> str:
    metrics = row.get("metrics") or {}
    bt = metrics.get("backtest") or {}
    candidate = {
        "tags": row.get("tags") or [],
        "parameters": row.get("parameters") or {},
        "entry_mode": (row.get("parameters") or {}).get("entry_mode") or "",
    }
    regime = row.get("regime") or row.get("regime_profile") or infer_regime_hint(candidate, bt)
    return str(regime or "unknown").strip().lower()


def _portfolio_score(row: dict[str, Any]) -> float:
    metrics = row.get("metrics") or {}
    agent = metrics.get("agent_score") or {}
    wf = metrics.get("walk_forward") or {}
    mc = metrics.get("monte_carlo") or {}
    perturb = metrics.get("perturbation") or {}
    bt = metrics.get("backtest") or {}

    agent_score = _safe_float(agent.get("score", 0.0), 0.0)
    wf_score = _safe_float(wf.get("score", 0.0), 0.0)
    mc_score = _safe_float(mc.get("score", 0.0), 0.0)
    perturb_score = _safe_float(perturb.get("score", 0.0), 0.0)
    ret = max(0.0, _safe_float(bt.get("return_pct", 0.0), 0.0))
    pf = max(0.0, _safe_float(bt.get("profit_factor", 0.0), 0.0))
    wr = max(0.0, _safe_float(bt.get("win_rate", 0.0), 0.0))
    dd = abs(_safe_float(bt.get("max_drawdown_pct", 0.0), 0.0))
    robustness = _safe_float(row.get("robustness_score", mc_score), 0.0)

    return (
        0.18 * agent_score
        + 0.16 * wf_score
        + 0.10 * mc_score
        + 0.10 * perturb_score
        + 0.16 * min(ret / 10.0, 1.0)
        + 0.10 * min(pf / 3.0, 1.0)
        + 0.10 * wr
        + 0.10 * max(0.0, 1.0 - min(dd / 20.0, 1.0))
        + 0.10 * robustness
    )


def _trade_pnls(row: dict[str, Any]) -> list[float]:
    metrics = row.get("metrics") or {}
    bt = metrics.get("backtest") or {}
    trades = bt.get("trades_detail") or []
    pnls: list[float] = []
    for trade in trades:
        if isinstance(trade, dict):
            pnls.append(_safe_float(trade.get("pnl", 0.0), 0.0))
    if pnls:
        return pnls
    avg = _safe_float(bt.get("avg_trade_pnl", 0.0), 0.0)
    n = int(bt.get("trades", 0) or 0)
    return [avg] * n if n > 0 else []


def _pearson_corr(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) < 2 or len(b) < 2:
        return 0.0
    n = min(len(a), len(b))
    x = list(a[:n])
    y = list(b[:n])
    mx = mean(x)
    my = mean(y)
    vx = sum((v - mx) ** 2 for v in x)
    vy = sum((v - my) ** 2 for v in y)
    if vx <= 0 or vy <= 0:
        return 0.0
    cov = sum((x[i] - mx) * (y[i] - my) for i in range(n))
    return max(-1.0, min(1.0, cov / sqrt(vx * vy)))


def _correlation_penalty(row: dict[str, Any], selected: Sequence[dict[str, Any]]) -> float:
    if not selected:
        return 0.0
    row_pnls = _trade_pnls(row)
    if not row_pnls:
        return 0.0

    penalties: list[float] = []
    row_symbol = _market_pair(row)[0].lower()
    row_regime = _regime_from_row(row)
    for other in selected:
        other_pnls = _trade_pnls(other)
        if not other_pnls:
            continue
        corr = abs(_pearson_corr(row_pnls, other_pnls))
        other_symbol = _market_pair(other)[0].lower()
        other_regime = _regime_from_row(other)
        structural = 0.0
        if row_symbol and row_symbol == other_symbol:
            structural += 0.15
        if row_regime and row_regime == other_regime:
            structural += 0.10
        penalties.append(min(0.55, corr * 0.45 + structural))
    if not penalties:
        return 0.0
    return max(0.0, min(0.65, mean(penalties)))


def select_portfolio_candidates(
    strategies: Sequence[dict[str, Any]],
    *,
    regime: str = "mean_reversion",
    limit: int = 3,
    unique_markets: bool = True,
) -> list[PortfolioCandidate]:
    regime = str(regime or "mean_reversion").strip().lower()
    candidates: list[PortfolioCandidate] = []
    seen: set[tuple[str, str]] = set()

    for row in strategies:
        status = str(row.get("status") or "").lower()
        if status not in {"deployable", "validated", "live"}:
            continue

        row_regime = _regime_from_row(row)
        if regime and regime not in {"all", "any"} and row_regime != regime:
            continue

        metrics = row.get("metrics") or {}
        bt = metrics.get("backtest") or {}
        if not bt:
            continue

        symbol, timeframe = _market_pair(row)
        key = (symbol, timeframe)
        if unique_markets and key in seen:
            continue
        seen.add(key)

        score = _portfolio_score(row)
        candidates.append(
            PortfolioCandidate(
                strategy_id=str(row.get("strategy_id") or ""),
                symbol=symbol,
                timeframe=timeframe,
                regime=row_regime,
                score=score,
                row=dict(row),
            )
        )

    candidates.sort(key=lambda c: (c.score, _safe_float((c.row.get("metrics") or {}).get("walk_forward", {}).get("score", 0.0), 0.0)), reverse=True)
    return candidates[: max(1, int(limit))]


def _normalize_weights(weights: Sequence[float] | None, n: int) -> list[float]:
    if n <= 0:
        return []
    if not weights:
        return [1.0 / n for _ in range(n)]
    vals = [max(0.0, float(w)) for w in weights[:n]]
    if len(vals) < n:
        vals.extend([0.0] * (n - len(vals)))
    total = sum(vals)
    if total <= 0:
        return [1.0 / n for _ in range(n)]
    return [v / total for v in vals]


def _trade_events(row: dict[str, Any], weight: float) -> list[dict[str, Any]]:
    metrics = row.get("metrics") or {}
    bt = metrics.get("backtest") or {}
    symbol = bt.get("symbol") or row.get("symbol") or "unknown"
    timeframe = bt.get("ltf_timeframe") or row.get("timeframe") or "unknown"
    trades = bt.get("trades_detail") or []
    if not trades:
        avg_trade = _safe_float(bt.get("avg_trade_pnl", 0.0), 0.0)
        trades_count = int(bt.get("trades", 0) or 0)
        if trades_count <= 0:
            return []
        trades = [{"ts": bt.get("ts") or bt.get("updated_at") or "1970-01-01T00:00:00+00:00", "pnl": avg_trade, "result": "AVG"} for _ in range(trades_count)]

    events: list[dict[str, Any]] = []
    for trade in trades:
        if not isinstance(trade, dict):
            continue
        pnl = _safe_float(trade.get("pnl", 0.0), 0.0)
        events.append(
            {
                "ts": _parse_time(trade.get("ts") or trade.get("timestamp") or trade.get("created_at")),
                "strategy_id": row.get("strategy_id"),
                "symbol": symbol,
                "timeframe": timeframe,
                "regime": _regime_from_row(row),
                "weight": float(weight),
                "base_pnl": pnl,
                "scaled_pnl": pnl * weight,
                "result": trade.get("result") or trade.get("reason") or "",
            }
        )
    return events


def evaluate_portfolio_combination(
    strategies: Sequence[dict[str, Any]],
    *,
    weights: Sequence[float] | None = None,
    total_capital: float = _BASE_EQUITY,
) -> PortfolioEvaluation:
    selected = list(strategies)
    if not selected:
        return PortfolioEvaluation(selected=[], weights=[], summary={"passed": False, "reason": "no_strategies"}, curve=[])

    normalized_weights = _normalize_weights(weights, len(selected))
    capital_scale = float(total_capital or _BASE_EQUITY) / _BASE_EQUITY

    events: list[dict[str, Any]] = []
    for row, weight in zip(selected, normalized_weights):
        row_events = _trade_events(row, weight=weight * capital_scale)
        events.extend(row_events)

    events.sort(key=lambda e: (e["ts"], str(e.get("strategy_id") or "")))

    equity = float(total_capital or _BASE_EQUITY)
    peak = equity
    worst_dd = 0.0
    curve: list[dict[str, Any]] = [{"ts": None, "equity": equity, "drawdown_pct": 0.0, "event": "start"}]
    gross_profit = 0.0
    gross_loss = 0.0
    wins = 0

    for event in events:
        pnl = _safe_float(event.get("scaled_pnl", 0.0), 0.0)
        equity += pnl
        if pnl > 0:
            gross_profit += pnl
            wins += 1
        elif pnl < 0:
            gross_loss += abs(pnl)
        peak = max(peak, equity)
        dd = ((equity - peak) / peak) * 100.0 if peak > 0 else 0.0
        worst_dd = min(worst_dd, dd)
        curve.append(
            {
                "ts": event["ts"].isoformat(),
                "equity": round(equity, 6),
                "drawdown_pct": round(dd, 6),
                "strategy_id": event.get("strategy_id"),
                "symbol": event.get("symbol"),
                "timeframe": event.get("timeframe"),
                "regime": event.get("regime"),
                "pnl": round(pnl, 6),
                "result": event.get("result"),
            }
        )

    total_trades = len(events)
    return_pct = ((equity / (total_capital or _BASE_EQUITY)) - 1.0) * 100.0 if total_capital else 0.0
    win_rate = wins / total_trades if total_trades else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else gross_profit if gross_profit > 0 else 0.0

    summary = {
        "passed": total_trades > 0 and return_pct > 0.0 and worst_dd > -15.0,
        "selected_count": len(selected),
        "total_trades": total_trades,
        "final_equity": round(equity, 6),
        "return_pct": round(return_pct, 6),
        "max_drawdown_pct": round(worst_dd, 6),
        "profit_factor": round(profit_factor, 6),
        "win_rate": round(win_rate, 6),
        "gross_profit": round(gross_profit, 6),
        "gross_loss": round(gross_loss, 6),
        "weights": [round(w, 6) for w in normalized_weights],
    }
    return PortfolioEvaluation(selected=[c.row if isinstance(c, PortfolioCandidate) else dict(c) for c in strategies], weights=list(normalized_weights), summary=summary, curve=curve)


def build_portfolio_summary(
    strategies: Sequence[dict[str, Any]],
    *,
    regime: str = "mean_reversion",
    limit: int = 3,
    unique_markets: bool = True,
    total_capital: float = _BASE_EQUITY,
) -> dict[str, Any]:
    selected = select_portfolio_candidates(strategies, regime=regime, limit=limit, unique_markets=unique_markets)
    if not selected:
        return {
            "regime": regime,
            "summary": {"passed": False, "reason": "no_eligible_strategies"},
            "selected": [],
            "curve": [],
        }

    weights = _normalize_weights(None, len(selected))
    evaluation = evaluate_portfolio_combination([c.row for c in selected], weights=weights, total_capital=total_capital)
    correlation_penalties = []
    for idx, candidate in enumerate(selected):
        penalty = _correlation_penalty(candidate.row, [c.row for c in selected[:idx]])
        correlation_penalties.append(round(penalty, 6))

    portfolio = {
        "regime": regime,
        "selected": [
            {
                "strategy_id": c.strategy_id,
                "symbol": c.symbol,
                "timeframe": c.timeframe,
                "regime": c.regime,
                "score": round(c.score, 6),
                "correlation_penalty": correlation_penalties[i],
            }
            for i, c in enumerate(selected)
        ],
        "summary": evaluation.summary,
        "curve": evaluation.curve,
    }
    return portfolio
