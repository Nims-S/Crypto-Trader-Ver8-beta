from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence

from config.execution import DEFAULT_TOTAL_CAPITAL


@dataclass(frozen=True)
class AllocationConstraints:
    total_capital: float
    temperature: float = 0.9
    cash_reserve_fraction: float = 0.20
    min_weight: float = 0.03
    max_weight: float = 0.35
    max_symbol_weight: float = 0.55
    max_regime_weight: float = 0.60
    min_allocatable_capital: float = 0.0


def _softmax(xs: Sequence[float], temperature: float = 1.0) -> List[float]:
    if not xs:
        return []
    t = max(1e-6, float(temperature or 1.0))
    scaled = [x / t for x in xs]
    m = max(scaled)
    exps = [math.exp(x - m) for x in scaled]
    s = sum(exps) or 1.0
    return [e / s for e in exps]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _norm(value: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    return max(0.0, min(1.0, (value - lo) / (hi - lo)))


def _extract_score_components(row: Mapping[str, Any]) -> dict[str, float]:
    metrics = row.get("metrics") or {}
    agent = metrics.get("agent_score") or {}
    wf = metrics.get("walk_forward") or {}
    bt = metrics.get("backtest") or {}
    mc = metrics.get("monte_carlo") or {}
    perturb = metrics.get("perturbation") or {}
    robustness = metrics.get("robustness") or {}

    backtest_return = _safe_float(bt.get("return_pct", 0.0), 0.0)
    backtest_pf = _safe_float(bt.get("profit_factor", 0.0), 0.0)
    backtest_wr = _safe_float(bt.get("win_rate", 0.0), 0.0)
    backtest_dd = abs(_safe_float(bt.get("max_drawdown_pct", 0.0), 0.0))
    backtest_trades = max(0.0, _safe_float(bt.get("trades", 0.0), 0.0))

    agent_score = _safe_float(agent.get("score", 0.0), 0.0)
    wf_score = _safe_float(wf.get("score", 0.0), 0.0)
    wf_passed = 1.0 if wf.get("passed") else 0.0
    robustness_score = _safe_float(
        row.get("robustness_score", robustness.get("score", 0.0) or perturb.get("score", 0.0)),
        0.0,
    )
    mc_score = _safe_float(mc.get("score", 0.0), 0.0)
    mc_passed = 1.0 if mc.get("passed") else 0.0
    perturb_score = _safe_float(perturb.get("score", 0.0), 0.0)
    perturb_passed = 1.0 if perturb.get("passed") else 0.0

    return {
        "agent": max(0.0, min(1.0, agent_score)),
        "wf": max(0.0, min(1.0, wf_score)),
        "wf_passed": wf_passed,
        "bt_return": _norm(backtest_return, -2.0, 12.0),
        "bt_pf": _norm(backtest_pf, 0.85, 4.5),
        "bt_wr": _norm(backtest_wr, 0.35, 0.72),
        "bt_dd": 1.0 - _norm(backtest_dd, 0.0, 20.0),
        "bt_trades": _norm(backtest_trades, 3.0, 30.0),
        "robustness": max(0.0, min(1.0, robustness_score)),
        "mc": max(0.0, min(1.0, mc_score)),
        "mc_passed": mc_passed,
        "perturb": max(0.0, min(1.0, perturb_score)),
        "perturb_passed": perturb_passed,
    }


def _regime_profile_tokens(row: Mapping[str, Any]) -> set[str]:
    tokens: set[str] = set()
    regime_profile = row.get("regime_profile")
    if isinstance(regime_profile, str) and regime_profile.strip():
        tokens.add(regime_profile.strip().lower())
    elif isinstance(regime_profile, (list, tuple, set)):
        tokens.update(str(x).strip().lower() for x in regime_profile if str(x).strip())

    tags = row.get("tags") or []
    for t in tags:
        token = str(t).strip().lower()
        if token:
            tokens.add(token)
    params = row.get("parameters") or {}
    entry_mode = str(params.get("entry_mode") or "").strip().lower()
    if entry_mode:
        tokens.add(entry_mode)
    return tokens


def _regime_match_factor(row: Mapping[str, Any], regime: str | None) -> float:
    if not regime:
        return 0.8
    regime = str(regime).strip().lower()
    if not regime:
        return 0.8
    tokens = _regime_profile_tokens(row)
    if regime in tokens:
        return 1.0
    if regime == "trend" and any(t in tokens for t in {"trend", "trend_following", "trend_pullback", "breakout"}):
        return 0.9
    if regime == "breakout" and any(t in tokens for t in {"breakout", "trend", "volatility"}):
        return 0.85
    if regime in {"mean_reversion", "mean-reversion", "mr", "chop"} and any(
        t in tokens for t in {"mean_reversion", "mean-reversion", "mr", "range", "chop"}
    ):
        return 0.92
    if regime == "volatility" and any(t in tokens for t in {"breakout", "volatility", "trend"}):
        return 0.8
    return 0.45


def _drift_factor(context: Mapping[str, Any] | None) -> float:
    if not context:
        return 1.0
    status = str(context.get("status") or "unknown").lower()
    multiplier = _safe_float(context.get("multiplier", 1.0), 1.0)
    if status == "disable":
        return 0.0
    if status == "throttle":
        return max(0.15, min(0.70, multiplier))
    if status == "warn":
        return max(0.40, min(0.85, multiplier))
    return max(0.25, min(1.0, multiplier))


def _diversification_factor(
    row: Mapping[str, Any],
    *,
    symbol_counts: Mapping[str, int] | None,
    regime_counts: Mapping[str, int] | None,
    allocated_strategies: Sequence[Mapping[str, Any]] | None,
) -> float:
    symbol = str(row.get("symbol") or row.get("market") or "").lower()
    regime = str(row.get("regime") or row.get("regime_profile") or "").lower()
    factor = 1.0

    if symbol and symbol_counts:
        factor *= 1.0 / math.sqrt(max(1, int(symbol_counts.get(symbol, 0)) + 1))
    if regime and regime_counts:
        factor *= 1.0 / math.sqrt(max(1, int(regime_counts.get(regime, 0)) + 1))

    if allocated_strategies:
        same_symbol = 0
        same_regime = 0
        for alloc in allocated_strategies:
            if str(alloc.get("symbol") or "").lower() == symbol and symbol:
                same_symbol += 1
            if str(alloc.get("regime") or "").lower() == regime and regime:
                same_regime += 1
        if same_symbol:
            factor *= 1.0 / (1.0 + 0.25 * same_symbol)
        if same_regime:
            factor *= 1.0 / (1.0 + 0.35 * same_regime)

    return max(0.30, min(1.0, factor))


def _portfolio_counts(context: Mapping[str, Any] | None) -> tuple[dict[str, int], dict[str, int], list[dict[str, Any]]]:
    symbol_counts: dict[str, int] = {}
    regime_counts: dict[str, int] = {}
    allocated: list[dict[str, Any]] = []
    if not context:
        return symbol_counts, regime_counts, allocated

    for item in context.get("allocated", []) or []:
        if not isinstance(item, dict):
            continue
        allocated.append(item)
        symbol = str(item.get("symbol") or "").lower()
        regime = str(item.get("regime") or "").lower()
        if symbol:
            symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1
        if regime:
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
    return symbol_counts, regime_counts, allocated


def _eligible(row: Mapping[str, Any], context: Mapping[str, Any] | None) -> bool:
    status = str(row.get("status") or "").lower()
    if status not in {"deployable", "validated", "live"}:
        return False

    metrics = row.get("metrics") or {}
    agent = metrics.get("agent_score") or {}
    wf = metrics.get("walk_forward") or {}
    mc = metrics.get("monte_carlo") or {}
    perturb = metrics.get("perturbation") or {}
    cross_symbol = metrics.get("cross_symbol") or {}

    # Hard gates: the allocator must never size capital for rows without full
    # robustness evidence. Missing evidence is treated as a failure.
    if not bool(agent.get("passed", False)):
        return False
    if not bool(wf.get("passed", False)):
        return False
    if not bool(mc.get("passed", False)):
        return False
    if not bool(perturb.get("passed", False)):
        return False
    if not bool(cross_symbol.get("passed", False)):
        return False

    if context:
        if not bool(context.get("enabled", True)):
            return False
        if str(context.get("status") or "").lower() == "disable":
            return False
    return True


def _score_row(
    row: Mapping[str, Any],
    *,
    context: Mapping[str, Any] | None,
    symbol_counts: Mapping[str, int] | None,
    regime_counts: Mapping[str, int] | None,
    allocated_strategies: Sequence[Mapping[str, Any]] | None,
) -> dict[str, float]:
    c = _extract_score_components(row)
    regime = str((context or {}).get("regime") or row.get("regime") or row.get("regime_profile") or "").lower()
    regime_match = _regime_match_factor(row, regime)
    drift = _drift_factor(context)
    diversification = _diversification_factor(
        row,
        symbol_counts=symbol_counts,
        regime_counts=regime_counts,
        allocated_strategies=allocated_strategies,
    )

    edge_score = (
        0.18 * c["agent"]
        + 0.14 * c["wf"]
        + 0.12 * c["bt_return"]
        + 0.12 * c["bt_pf"]
        + 0.12 * c["bt_wr"]
        + 0.14 * c["bt_dd"]
        + 0.08 * c["bt_trades"]
        + 0.10 * c["robustness"]
        + 0.05 * c["mc"]
        + 0.05 * c["perturb"]
    )
    final_score = edge_score * regime_match * drift * diversification

    return {
        "edge": max(0.0, min(1.0, edge_score)),
        "regime_match": regime_match,
        "drift": drift,
        "diversification": diversification,
        "score": max(0.0, min(1.0, final_score)),
        "raw": c,
    }


def _cap_weights(
    strategies: Sequence[Mapping[str, Any]],
    scores: Sequence[float],
    total_capital: float,
    *,
    context: Mapping[str, Any] | None,
    constraints: AllocationConstraints,
) -> list[float]:
    if not strategies or total_capital <= 0 or not scores or max(scores) <= 0:
        return [0.0 for _ in strategies]

    allocatable_capital = max(0.0, total_capital * (1.0 - constraints.cash_reserve_fraction))
    weights = _softmax(scores, temperature=constraints.temperature)
    capitals = [allocatable_capital * w for w in weights]

    symbol_budgets: dict[str, float] = {}
    regime_budgets: dict[str, float] = {}
    if context:
        symbol_budgets = {str(k).lower(): _safe_float(v, 0.0) for k, v in (context.get("symbol_caps") or {}).items()}
        regime_budgets = {str(k).lower(): _safe_float(v, 0.0) for k, v in (context.get("regime_caps") or {}).items()}

    for _ in range(8):
        capped = [False] * len(capitals)
        for i, row in enumerate(strategies):
            sid = str(row.get("strategy_id") or "")
            symbol = str(row.get("symbol") or row.get("market") or "").lower()
            regime = str((context or {}).get("regime") or row.get("regime") or row.get("regime_profile") or "").lower()
            max_cap = None

            row_ctx = (context or {}).get(sid, {}) if context else {}
            if row_ctx:
                max_cap = _safe_float(row_ctx.get("max_capital", 0.0), 0.0) or None
            if max_cap is None and symbol in symbol_budgets:
                max_cap = symbol_budgets[symbol]
            if max_cap is None and regime in regime_budgets:
                max_cap = regime_budgets[regime]
            if max_cap is None:
                max_cap = allocatable_capital * constraints.max_weight

            if max_cap >= 0.0 and capitals[i] > max_cap:
                capitals[i] = max_cap
                capped[i] = True

        remaining = max(0.0, allocatable_capital - sum(capitals))
        if remaining <= 1e-9:
            break

        uncapped_idx = [i for i, was_capped in enumerate(capped) if not was_capped]
        if not uncapped_idx:
            break
        uncapped_scores = [scores[i] for i in uncapped_idx]
        if not uncapped_scores or max(uncapped_scores) <= 0:
            break
        sub_weights = _softmax(uncapped_scores, temperature=constraints.temperature)
        for w, i in zip(sub_weights, uncapped_idx):
            capitals[i] += remaining * w

    total_allocated = sum(capitals)
    if total_allocated <= 0:
        return [0.0 for _ in strategies]
    weights = [c / total_allocated for c in capitals]

    weights = [max(constraints.min_weight, min(constraints.max_weight, w)) for w in weights]
    s = sum(weights) or 1.0
    return [w / s for w in weights]


def allocate_capital(
    strategies: List[dict[str, Any]],
    total_capital: float = DEFAULT_TOTAL_CAPITAL,
    *,
    temperature: float = 0.9,
    min_weight: float = 0.03,
    max_weight: float = 0.35,
    cash_reserve_fraction: float = 0.20,
    context: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    if not strategies:
        return []

    constraints = AllocationConstraints(
        total_capital=float(total_capital or DEFAULT_TOTAL_CAPITAL),
        temperature=float(temperature or 0.9),
        cash_reserve_fraction=float(cash_reserve_fraction or 0.0),
        min_weight=float(min_weight or 0.0),
        max_weight=float(max_weight or 1.0),
    )

    symbol_counts, regime_counts, allocated = _portfolio_counts(context)
    scored_rows: list[dict[str, Any]] = []
    scores: list[float] = []

    for row in strategies:
        row = dict(row)
        sid = str(row.get("strategy_id") or "")
        row_context = (context or {}).get(sid, {}) if context else {}
        if not _eligible(row, row_context):
            row["allocation_score"] = 0.0
            row["allocation_reasons"] = ["ineligible"]
            scored_rows.append(row)
            scores.append(0.0)
            continue

        regime_name = row_context.get("regime") or row.get("regime") or row.get("regime_profile")
        if regime_name:
            row["regime"] = regime_name

        scoring = _score_row(
            row,
            context=row_context,
            symbol_counts=symbol_counts,
            regime_counts=regime_counts,
            allocated_strategies=allocated,
        )
        row["allocation_score"] = scoring["score"]
        row["allocation_edge"] = scoring["edge"]
        row["allocation_components"] = scoring
        row["regime"] = str(row.get("regime") or row_context.get("regime") or row.get("regime_profile") or "").lower() or None
        row["symbol"] = row.get("symbol") or row.get("market")
        row["timeframe"] = row.get("timeframe")
        row["capital_floor"] = _safe_float(row_context.get("min_capital", 0.0), 0.0)
        row["capital_ceiling"] = _safe_float(row_context.get("max_capital", 0.0), 0.0)
        scored_rows.append(row)
        scores.append(scoring["score"])

    weights = _cap_weights(
        scored_rows,
        scores,
        constraints.total_capital,
        context=context,
        constraints=constraints,
    )

    allocatable_capital = max(0.0, constraints.total_capital * (1.0 - constraints.cash_reserve_fraction))
    allocations: list[Dict[str, Any]] = []
    for row, weight, score in zip(scored_rows, weights, scores):
        capital = allocatable_capital * weight if score > 0 else 0.0
        floor_cap = _safe_float(row.get("capital_floor", 0.0), 0.0)
        ceiling_cap = _safe_float(row.get("capital_ceiling", 0.0), 0.0)
        if floor_cap > 0:
            capital = max(capital, floor_cap)
        if ceiling_cap > 0:
            capital = min(capital, ceiling_cap)
        if capital < constraints.min_allocatable_capital:
            capital = 0.0

        allocations.append(
            {
                "strategy_id": row.get("strategy_id"),
                "symbol": row.get("symbol"),
                "timeframe": row.get("timeframe"),
                "regime": row.get("regime"),
                "weight": float(weight),
                "capital": float(capital),
                "score": float(score),
                "allocation_score": float(row.get("allocation_score", 0.0) or 0.0),
                "allocation_edge": float(row.get("allocation_edge", 0.0) or 0.0),
                "allocation_components": row.get("allocation_components") or {},
                "status": row.get("status"),
            }
        )

    total = sum(a["capital"] for a in allocations)
    if total > 0:
        for alloc in allocations:
            alloc["weight"] = alloc["capital"] / total
    return allocations
