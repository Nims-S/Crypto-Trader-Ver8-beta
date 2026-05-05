from __future__ import annotations

from typing import Any


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _compact_list(values: Any, limit: int = 4) -> list[str]:
    out: list[str] = []
    if not isinstance(values, (list, tuple)):
        return out
    for value in values[: max(0, int(limit))]:
        text = str(value).strip()
        if text:
            out.append(text)
    return out


def _jsonish_lines(items: list[tuple[str, Any]]) -> str:
    parts: list[str] = []
    for key, value in items:
        parts.append(f'"{key}": {value!r}')
    return "{ " + ", ".join(parts) + " }"


def _normalize_objective(context: dict[str, Any]) -> str:
    objective = str(context.get("objective_metric") or "").strip().lower()
    if objective:
        return objective
    directives = context.get("mutation_directives") or {}
    if directives.get("loosen_filters"):
        return "trade_density"
    if directives.get("tighten_exits"):
        return "profit_factor"
    if directives.get("shorten_holding"):
        return "drawdown"
    if directives.get("prefer_breakout"):
        return "trade_density"
    return "walk_forward_stability"


def _normalize_target_regime(context: dict[str, Any]) -> str:
    target = str(context.get("target_regime") or "").strip().lower()
    if target:
        return target
    directives = context.get("mutation_directives") or {}
    if directives.get("prefer_breakout"):
        return "breakout"
    if directives.get("prefer_trend_pullback"):
        return "trend"
    if directives.get("prefer_structure"):
        return "trend"
    return "balanced"


def build_hermes_prompt(context: dict[str, Any]) -> str:
    symbol = str(context.get("symbol") or "BTC/USDT")
    timeframe = str(context.get("timeframe") or "1h")
    failure = context.get("failure_profile") or {}
    trade_activity = context.get("trade_activity") or {}
    directives = context.get("mutation_directives") or {}
    objective = _normalize_objective(context)
    target_regime = _normalize_target_regime(context)

    counts = failure.get("counts") or {}
    trade_mean = trade_activity.get("mean") or {}
    pf_mean = trade_activity.get("mean_pf") or {}
    wr_mean = trade_activity.get("mean_wr") or {}
    notes = _compact_list(failure.get("notes"), limit=5)

    payload = _jsonish_lines(
        [
            ("symbol", symbol),
            ("timeframe", timeframe),
            ("objective_metric", objective),
            ("target_regime", target_regime),
            ("primary_failure", failure.get("primary", "other")),
            ("counts", counts),
            ("trade_mean", trade_mean),
            ("pf_mean", pf_mean),
            ("wr_mean", wr_mean),
            ("top_failure_notes", notes),
            ("directives", directives),
        ]
    )

    return (
        "You are Hermes, a trading strategy hypothesis generator.\n"
        "Return exactly 3 hypothesis packets as valid JSON array and nothing else.\n"
        "Each packet must include: hypothesis_id, parent_strategy_id, target_regime, strategy_family, "
        "objective_metric, expected_improvement, failure_modes, entry_ideas, exit_ideas, volatility_adaptation, "
        "trade_density_expectation, robustness_checks, gating_rules, notes.\n"
        "Constraints: do not propose random indicator spam; keep the packets meaningfully different; at least one packet "
        "must optimize density, one must optimize stability, and one must optimize drawdown.\n"
        f"Context: {payload}"
    )


def build_claude_prompt(context: dict[str, Any]) -> str:
    symbol = str(context.get("symbol") or "BTC/USDT")
    timeframe = str(context.get("timeframe") or "1h")
    parent_id = str(context.get("parent_strategy_id") or "seed")
    objective = _normalize_objective(context)
    target_regime = _normalize_target_regime(context)
    directives = context.get("mutation_directives") or {}
    failure = context.get("failure_profile") or {}
    top_notes = _compact_list(failure.get("notes"), limit=5)

    payload = _jsonish_lines(
        [
            ("parent_strategy_id", parent_id),
            ("symbol", symbol),
            ("timeframe", timeframe),
            ("objective_metric", objective),
            ("target_regime", target_regime),
            ("failure_profile", failure),
            ("top_failure_notes", top_notes),
            ("directives", directives),
        ]
    )

    return (
        "You are Claude Code inside a trading research pipeline.\n"
        "Mutate the parent strategy into exactly 3 children.\n"
        "Return valid JSON only with keys: parent_strategy_id, children.\n"
        "Each child must include: child_strategy_id, mutations, parameter_updates, expected_effect, target_regime, failure_mode, objective_metric.\n"
        "Keep changes minimal and traceable. Do not introduce unrelated indicators. Preserve core risk management.\n"
        "At least one child must prioritize density, one stability, and one drawdown reduction.\n"
        f"Context: {payload}"
    )


def build_child_batch_prompts(context: dict[str, Any], n: int = 3) -> list[dict[str, str]]:
    """Create diverse prompts for parallel mutation."""
    objective = _normalize_objective(context)
    target_regime = _normalize_target_regime(context)
    failure = context.get("failure_profile") or {}
    top_notes = _compact_list(failure.get("notes"), limit=5)

    goals = [
        ("density", "increase trade density without blowing up drawdown"),
        ("stability", "improve walk-forward consistency and reduce split spread"),
        ("drawdown", "reduce drawdown and equity-curve volatility"),
    ]
    outputs: list[dict[str, str]] = []
    for i in range(max(1, n)):
        goal_name, goal_desc = goals[i % len(goals)]
        prompt = (
            "You are Claude Code. Mutate the strategy with a narrow, targeted change set.\n"
            "Return valid JSON only with keys: child_strategy_id, parameter_updates, rationale, target_regime, objective_metric, failure_mode, expected_effect.\n"
            f"Primary objective: {goal_name}.\n"
            f"Objective detail: {goal_desc}.\n"
            f"Global objective_metric: {objective}. Target regime: {target_regime}.\n"
            f"Known failure notes: {top_notes}.\n"
            "Rules: keep the parent logic recognizable; do not spam indicators; prefer parameter and filter changes over framework rewrites; "
            "make this child structurally distinct from the other children.\n"
            f"Context: {context}"
        )
        outputs.append({"goal": goal_name, "prompt": prompt})
    return outputs


def build_validator_prompt(context: dict[str, Any]) -> str:
    symbol = str(context.get("symbol") or "BTC/USDT")
    timeframe = str(context.get("timeframe") or "1h")
    metrics = context.get("metrics") or {}
    status_ladder = ["candidate", "validated", "deployable", "live", "rejected"]
    failure = context.get("failure_profile") or {}
    top_notes = _compact_list(failure.get("notes"), limit=5)

    payload = _jsonish_lines(
        [
            ("symbol", symbol),
            ("timeframe", timeframe),
            ("objective_metric", _normalize_objective(context)),
            ("target_regime", _normalize_target_regime(context)),
            ("status_ladder", status_ladder),
            ("metrics", metrics),
            ("failure_profile", failure),
            ("top_failure_notes", top_notes),
        ]
    )

    return (
        "You are a strategy validator. Return valid JSON only.\n"
        "Decide one status from candidate, validated, deployable, live, rejected.\n"
        "Use robustness and walk-forward stability first, then density, then drawdown, then profitability.\n"
        "Return keys: status, score, deployable, quality, reasons, notes.\n"
        "Reject if train/val/test diverge, if density is too low for the regime, or if drawdown is excessive.\n"
        f"Context: {payload}"
    )


def build_prompt_bundle(context: dict[str, Any]) -> dict[str, str]:
    return {
        "hermes": build_hermes_prompt(context),
        "claude": build_claude_prompt(context),
        "validator": build_validator_prompt(context),
    }
