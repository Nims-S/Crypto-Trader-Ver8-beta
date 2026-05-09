from __future__ import annotations

from typing import Any, Sequence

from research.portfolio import evaluate_portfolio_combination


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _market_pair(row: dict[str, Any]) -> tuple[str, str]:
    metrics = row.get("metrics") or {}
    bt = metrics.get("backtest") or {}
    symbol = str(bt.get("symbol") or row.get("symbol") or row.get("market") or "unknown")
    timeframe = str(bt.get("ltf_timeframe") or row.get("timeframe") or bt.get("timeframe") or "unknown")
    return symbol, timeframe


def _regime_from_row(row: dict[str, Any]) -> str:
    regime = row.get("regime") or row.get("regime_profile") or row.get("status") or "unknown"
    return str(regime).strip().lower() or "unknown"


def _soft_fill_eligible(row: dict[str, Any]) -> bool:
    status = str(row.get("status") or "").lower()
    if status not in {"validated", "deployable", "live"}:
        return False

    metrics = row.get("metrics") or {}
    agent = metrics.get("agent_score") or {}
    wf = metrics.get("walk_forward") or {}
    bt = metrics.get("backtest") or {}
    mc = metrics.get("monte_carlo")
    perturb = metrics.get("perturbation")

    if not bool(agent.get("passed", False)):
        return False
    if not bool(wf.get("passed", False)):
        return False
    if mc is None or perturb is None:
        return False

    trades = max(0, int(bt.get("trades", 0) or 0))
    ret = _safe_float(bt.get("return_pct", 0.0), 0.0)
    pf = _safe_float(bt.get("profit_factor", 0.0), 0.0)
    wr = _safe_float(bt.get("win_rate", 0.0), 0.0)
    dd = abs(_safe_float(bt.get("max_drawdown_pct", 0.0), 0.0))
    robustness = _safe_float(row.get("robustness_score", 0.0), 0.0)

    if trades < 4:
        return False
    if ret < 0.0 or pf < 1.05 or wr < 0.40:
        return False
    if dd > 10.0:
        return False
    if robustness < 0.45:
        return False
    return True


def _soft_fill_score(row: dict[str, Any]) -> float:
    metrics = row.get("metrics") or {}
    agent = metrics.get("agent_score") or {}
    wf = metrics.get("walk_forward") or {}
    bt = metrics.get("backtest") or {}
    mc = metrics.get("monte_carlo") or {}
    perturb = metrics.get("perturbation") or {}

    agent_score = _safe_float(agent.get("score", 0.0), 0.0)
    wf_score = _safe_float(wf.get("score", 0.0), 0.0)
    bt_return = max(0.0, _safe_float(bt.get("return_pct", 0.0), 0.0))
    bt_pf = max(0.0, _safe_float(bt.get("profit_factor", 0.0), 0.0))
    wr = max(0.0, _safe_float(bt.get("win_rate", 0.0), 0.0))
    dd = abs(_safe_float(bt.get("max_drawdown_pct", 0.0), 0.0))
    robustness = _safe_float(row.get("robustness_score", 0.0), 0.0)
    mc_score = _safe_float(mc.get("score", 0.0), 0.0) if isinstance(mc, dict) else 0.0
    perturb_score = _safe_float(perturb.get("score", 0.0), 0.0) if isinstance(perturb, dict) else 0.0

    return (
        0.22 * agent_score
        + 0.16 * wf_score
        + 0.14 * robustness
        + 0.12 * min(bt_return / 10.0, 1.0)
        + 0.12 * min(bt_pf / 3.0, 1.0)
        + 0.10 * wr
        + 0.10 * max(0.0, 1.0 - min(dd / 10.0, 1.0))
        + 0.02 * mc_score
        + 0.02 * perturb_score
    )


def _normalize_weights(scores: Sequence[float]) -> list[float]:
    if not scores:
        return []
    vals = [max(0.0, float(s)) for s in scores]
    total = sum(vals)
    if total <= 0:
        return [1.0 / len(vals) for _ in vals]
    return [v / total for v in vals]


def select_soft_fill_candidates(
    strategies: Sequence[dict[str, Any]],
    *,
    regime: str = "mean_reversion",
    limit: int = 3,
    unique_markets: bool = True,
) -> list[dict[str, Any]]:
    regime = str(regime or "mean_reversion").strip().lower()
    selected: list[dict[str, Any]] = []
    seen_markets: set[tuple[str, str]] = set()

    scored: list[dict[str, Any]] = []
    for row in strategies:
        if not _soft_fill_eligible(row):
            continue
        row_regime = _regime_from_row(row)
        if regime not in {"all", "any"} and regime and regime != row_regime:
            # Soft fill is intended for mean-reversion probationary baskets, but
            # still allows regime-compatible rows when called with another target.
            if regime != "mean_reversion":
                continue
        symbol, timeframe = _market_pair(row)
        score = _soft_fill_score(row)
        scored.append(
            {
                "strategy_id": str(row.get("strategy_id") or ""),
                "symbol": symbol,
                "timeframe": timeframe,
                "regime": row_regime,
                "score": score,
                "raw_score": score,
                "row": dict(row),
            }
        )

    scored.sort(key=lambda r: (r["score"], _safe_float((r["row"].get("robustness_score") if isinstance(r.get("row"), dict) else 0.0), 0.0)), reverse=True)
    for cand in scored:
        if len(selected) >= max(1, int(limit)):
            break
        market_key = (cand["symbol"].lower(), cand["timeframe"].lower())
        if unique_markets and market_key in seen_markets:
            continue
        selected.append(cand)
        seen_markets.add(market_key)

    return selected


def build_soft_fill_portfolio_summary(
    strategies: Sequence[dict[str, Any]],
    *,
    regime: str = "mean_reversion",
    limit: int = 3,
    unique_markets: bool = True,
    total_capital: float = 10_000.0,
    probationary_capital_fraction: float = 0.35,
) -> dict[str, Any]:
    selected = select_soft_fill_candidates(
        strategies,
        regime=regime,
        limit=limit,
        unique_markets=unique_markets,
    )
    if not selected:
        return {
            "regime": regime,
            "selected": [],
            "weights": [],
            "curve": [],
            "summary": {
                "passed": False,
                "reason": "no_eligible_strategies",
                "soft_fill": True,
                "probationary": False,
                "probationary_capital": 0.0,
            },
        }

    probationary_fraction = max(0.05, min(1.0, float(probationary_capital_fraction or 0.35)))
    probationary_capital = float(total_capital or 10_000.0) * probationary_fraction
    weights = _normalize_weights([cand["score"] for cand in selected])
    evaluation = evaluate_portfolio_combination([cand["row"] for cand in selected], weights=weights, total_capital=probationary_capital)
    summary = dict(evaluation.summary)
    summary.update(
        {
            "soft_fill": True,
            "probationary": True,
            "probationary_capital": round(probationary_capital, 6),
            "probationary_capital_fraction": round(probationary_fraction, 6),
            "selected_count": len(selected),
            "passed": bool(summary.get("passed", False)) and len(selected) >= 2,
        }
    )

    return {
        "regime": regime,
        "selected": selected,
        "weights": weights,
        "summary": summary,
        "curve": evaluation.curve,
    }
