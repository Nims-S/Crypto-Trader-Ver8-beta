from __future__ import annotations

import math
from typing import Any, List, Dict


def _softmax(xs: List[float], temperature: float = 1.0) -> List[float]:
    if not xs:
        return []
    t = max(1e-6, float(temperature or 1.0))
    scaled = [x / t for x in xs]
    m = max(scaled)
    exps = [math.exp(x - m) for x in scaled]
    s = sum(exps) or 1.0
    return [e / s for e in exps]


def _score_row(row: dict[str, Any]) -> float:
    wf = (row.get("metrics") or {}).get("walk_forward") or {}
    base = float(wf.get("score", 0.0) or 0.0)
    robustness = float(row.get("robustness_score", 0.0) or 0.0)
    return base * (1.0 + 0.3 * robustness)


def _apply_caps(
    strategies: List[dict[str, Any]],
    scores: List[float],
    total_capital: float,
    context: Dict[str, dict[str, Any]] | None,
    temperature: float,
) -> List[float]:
    if total_capital <= 0 or not scores or max(scores) <= 0:
        return [0.0 for _ in strategies]

    weights = _softmax(scores, temperature=temperature)
    capitals = [total_capital * w for w in weights]

    for _ in range(5):
        capped = [False] * len(capitals)
        for i, row in enumerate(strategies):
            sid = row.get("strategy_id")
            cap = None
            if context and sid in context:
                cap = float(context[sid].get("max_capital", 0.0) or 0.0)
            if cap is not None and cap >= 0.0 and capitals[i] > cap:
                capitals[i] = cap
                capped[i] = True

        remaining = max(0.0, total_capital - sum(capitals))
        if remaining <= 1e-9:
            break

        uncapped_idx = [i for i, c in enumerate(capped) if not c]
        if not uncapped_idx:
            break
        uncapped_scores = [scores[i] for i in uncapped_idx]
        if not uncapped_scores or max(uncapped_scores) <= 0:
            break
        sub_weights = _softmax(uncapped_scores, temperature=temperature)
        for w, i in zip(sub_weights, uncapped_idx):
            capitals[i] += remaining * w

    return capitals


def allocate_capital(
    strategies: List[dict[str, Any]],
    total_capital: float,
    *,
    temperature: float = 1.0,
    min_weight: float = 0.0,
    context: Dict[str, dict] | None = None,
) -> List[Dict[str, Any]]:
    if not strategies:
        return []

    scores = []
    for r in strategies:
        sid = r.get("strategy_id")
        base = _score_row(r)
        ctx = (context or {}).get(sid, {})
        mult = float(ctx.get("multiplier", 1.0))
        enabled = bool(ctx.get("enabled", True))
        score = base * mult if enabled else 0.0
        scores.append(score)

    capitals = _apply_caps(strategies, scores, float(total_capital or 0.0), context, temperature)
    total = sum(capitals) or 1.0
    weights = [c / total for c in capitals]

    if min_weight > 0:
        weights = [max(min_weight, w) for w in weights]
        s = sum(weights) or 1.0
        weights = [w / s for w in weights]

    allocations = []
    for row, w, c in zip(strategies, weights, capitals):
        allocations.append(
            {
                "strategy_id": row.get("strategy_id"),
                "weight": float(w),
                "capital": float(c),
                "score": _score_row(row),
            }
        )
    return allocations
