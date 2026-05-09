from __future__ import annotations

import copy
import hashlib
import json
import random
from dataclasses import dataclass
from typing import Any, Dict, List

from research.feedback import build_feedback_summary
from research.llm_batch import PromptJob, batch_prompts_sync
from research.prompt_templates import build_child_batch_prompts
from research.llm_client import get_default_llm_client


@dataclass
class StrategyCandidate:
    strategy_id: str
    base_strategy: str
    version: int
    parameters: Dict[str, Any]
    symbol: str
    timeframe: str
    tags: list
    source: str
    notes: str = ""


def _safe_float(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return default


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _signature(params: dict) -> str:
    try:
        s = json.dumps(params, sort_keys=True)
    except Exception:
        s = str(params)
    return hashlib.sha1(s.encode()).hexdigest()[:16]


def _distance(a: dict, b: dict) -> float:
    keys = set(a) | set(b)
    if not keys:
        return 1.0
    total = 0.0
    for k in keys:
        va, vb = a.get(k), b.get(k)
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            total += abs(float(va) - float(vb)) / (abs(float(va)) + abs(float(vb)) + 1e-6)
        else:
            total += 0.0 if va == vb else 1.0
    return total / len(keys)


def _objective_from_feedback(feedback: dict) -> str:
    profile = (feedback or {}).get("failure_profile") or {}
    primary = str(profile.get("primary") or "")
    trade_mean = ((feedback or {}).get("trade_activity") or {}).get("mean", {}) or {}
    test_trades = _safe_float(trade_mean.get("test", 0.0), 0.0)
    quality_floor = bool((feedback or {}).get("quality_floor_passed", False))

    if primary == "no_trades" or test_trades < 12:
        return "density"
    if primary == "low_profit_factor":
        return "profit_factor"
    if primary in {"unstable", "high_drawdown"}:
        return "stability"
    if not quality_floor:
        return "density"
    return "balanced"


def _candidate_priority(params: dict, objective: str) -> float:
    cooldown = _safe_float(params.get("cooldown_bars", 24), 24)
    max_bars = _safe_float(params.get("max_bars_override", 60), 60)
    filter_count = sum(
        1
        for k in (
            "use_htf_filter",
            "use_volume_filter",
            "use_structure_filter",
            "use_reclaim_filter",
            "use_trend_filter",
            "use_breakout_filter",
        )
        if params.get(k)
    )
    mode = str(params.get("entry_mode") or "")
    tp1 = _safe_float(params.get("tp1_rr", 2.0), 2.0)
    tp2 = _safe_float(params.get("tp2_rr", max(tp1 + 0.75, 3.0)), 3.0)
    stop_mult = _safe_float(params.get("stop_atr_mult", 2.0), 2.0)

    score = 0.0
    if objective == "density":
        score += (1.0 / (1.0 + cooldown)) * 2.5
        score += (1.0 / (1.0 + max_bars)) * 1.2
        score += (1.0 / (1.0 + filter_count)) * 1.0
        if mode in {"breakout", "mean_reversion", "trend_pullback"}:
            score += 0.8
        score += 0.15 * (2.5 - min(tp1, 3.5))
    elif objective == "stability":
        score += (filter_count / 6.0) * 1.5
        score += _safe_float(params.get("min_adx", 10), 10) / 40.0
        score += max(0.0, 2.5 - stop_mult) * 0.15
    elif objective == "drawdown":
        score += 1.0 / (1.0 + _safe_float(params.get("size_multiplier", 1.0), 1.0))
        score += _safe_float(params.get("stop_atr_mult", 1.5), 1.5) / 3.0
    elif objective == "profit_factor":
        score += tp1 / 5.0
        score += tp2 / 8.0
        score += (1.0 / (1.0 + cooldown))
    else:
        score += 0.5 * (1.0 / (1.0 + cooldown))
        score += 0.5 * (1.0 / (1.0 + max_bars))
    return float(score)


def _apply_directives(params: dict, directives: dict, symbol: str) -> dict:
    if not directives:
        return params

    for key in (
        "use_htf_filter",
        "use_volume_filter",
        "use_structure_filter",
        "use_reclaim_filter",
        "use_trend_filter",
        "use_breakout_filter",
        "entry_mode",
    ):
        if key in directives:
            params[key] = directives[key]

    for key in (
        "min_adx",
        "min_bb_rank",
        "min_atr_rank",
        "htf_adx_min",
        "htf_bb_rank_min",
        "rsi_min",
        "rsi_max",
        "volume_multiplier",
        "pullback_lookback",
        "pullback_bars",
        "stop_atr_mult",
        "tp1_rr",
        "tp2_rr",
        "tp1_close_fraction",
        "tp2_close_fraction",
        "tp3_close_fraction",
        "trail_atr_mult",
        "trail_ema20",
        "cooldown_bars",
        "max_bars_override",
        "confidence",
        "size_multiplier",
        "be_trigger_rr",
    ):
        if key in directives:
            params[key] = directives[key]

    if symbol.startswith("BTC") and directives.get("prefer_trend_pullback"):
        params["entry_mode"] = "trend_pullback"
        params["use_htf_filter"] = True
        params["use_trend_filter"] = True

    return params


def _apply_safety_constraints(params: dict, objective: str, symbol: str) -> dict:
    p = dict(params or {})
    mode = str(p.get("entry_mode") or "").strip().lower()

    if mode in {"trend", "trend_following"}:
        p["entry_mode"] = "trend_pullback"
    elif mode not in {"trend_pullback", "breakout", "mean_reversion"}:
        p["entry_mode"] = "trend_pullback" if symbol.startswith("BTC") else "breakout"

    p["stop_atr_mult"] = _clamp(_safe_float(p.get("stop_atr_mult", 2.0), 2.0), 1.2, 3.5)
    p["tp1_rr"] = _clamp(_safe_float(p.get("tp1_rr", 2.2), 2.2), 1.4, 4.5)
    p["tp2_rr"] = _clamp(_safe_float(p.get("tp2_rr", max(p["tp1_rr"] + 0.8, 3.2)), 3.2), p["tp1_rr"] + 0.5, 7.0)

    cd_lo, cd_hi = (6, 18) if objective == "density" else (8, 28)
    p["cooldown_bars"] = int(_clamp(int(_safe_float(p.get("cooldown_bars", 18), 18)), cd_lo, cd_hi))
    p["pullback_bars"] = int(_clamp(int(_safe_float(p.get("pullback_bars", 1), 1)), 1, 4))
    p["pullback_lookback"] = int(_clamp(int(_safe_float(p.get("pullback_lookback", 3), 3)), 2, 6))

    adx_hi = 24.0 if objective == "density" else 30.0
    bb_hi = 0.45 if objective == "density" else 0.60
    atr_hi = 0.40 if objective == "density" else 0.60
    p["min_adx"] = _clamp(_safe_float(p.get("min_adx", 20.0), 20.0), 12.0, adx_hi)
    p["min_bb_rank"] = _clamp(_safe_float(p.get("min_bb_rank", 0.28), 0.28), 0.08, bb_hi)
    p["min_atr_rank"] = _clamp(_safe_float(p.get("min_atr_rank", 0.22), 0.22), 0.08, atr_hi)

    p["htf_adx_min"] = _clamp(_safe_float(p.get("htf_adx_min", 18.0), 18.0), 10.0, 28.0)
    p["htf_bb_rank_min"] = _clamp(_safe_float(p.get("htf_bb_rank_min", 0.20), 0.20), 0.05, 0.50)
    p["rsi_min"] = _clamp(_safe_float(p.get("rsi_min", 50.0), 50.0), 40.0, 60.0)
    p["rsi_max"] = _clamp(_safe_float(p.get("rsi_max", 68.0), 68.0), 55.0, 80.0)
    p["volume_multiplier"] = _clamp(_safe_float(p.get("volume_multiplier", 1.02), 1.02), 0.80, 1.30)
    p["max_bars_override"] = int(_clamp(int(_safe_float(p.get("max_bars_override", 60), 60)), 18, 144))
    p["confidence"] = _clamp(_safe_float(p.get("confidence", 0.75), 0.75), 0.45, 0.95)
    p["size_multiplier"] = _clamp(_safe_float(p.get("size_multiplier", 0.80), 0.80), 0.25, 1.00)

    if objective == "density":
        p["use_volume_filter"] = bool(p.get("use_volume_filter", False)) and random.random() < 0.70
        p["use_structure_filter"] = bool(p.get("use_structure_filter", False)) and random.random() < 0.75
        p["use_breakout_filter"] = bool(p.get("use_breakout_filter", False)) or p["entry_mode"] == "breakout"
        p["volume_multiplier"] = _clamp(p["volume_multiplier"] - 0.03, 0.80, 1.20)
        p["min_adx"] = min(p["min_adx"], 22.0)
        p["cooldown_bars"] = min(p["cooldown_bars"], 16)

    return p


def _baseline_params(symbol: str, objective: str) -> dict:
    if symbol.startswith("BTC"):
        params = {
            "entry_mode": "trend_pullback",
            "use_htf_filter": True,
            "use_volume_filter": True,
            "use_structure_filter": True,
            "use_trend_filter": True,
            "min_adx": 19.0,
            "htf_adx_min": 16.0,
            "htf_bb_rank_min": 0.24,
            "min_bb_rank": 0.26,
            "min_atr_rank": 0.22,
            "rsi_min": 50.0,
            "rsi_max": 68.0,
            "volume_multiplier": 1.02,
            "pullback_lookback": 3,
            "pullback_bars": 1,
            "stop_atr_mult": 1.8,
            "tp1_rr": 2.0,
            "tp2_rr": 3.2,
            "tp1_close_fraction": 0.45,
            "tp2_close_fraction": 0.55,
            "trail_atr_mult": 1.3,
            "cooldown_bars": 14,
            "max_bars_override": 72,
            "confidence": 0.80,
            "size_multiplier": 0.85,
        }
    else:
        params = {
            "entry_mode": "breakout",
            "use_htf_filter": True,
            "use_volume_filter": True,
            "use_breakout_filter": True,
            "min_adx": 17.0,
            "min_bb_rank": 0.22,
            "min_atr_rank": 0.18,
            "stop_atr_mult": 1.6,
            "tp1_rr": 1.9,
            "tp2_rr": 3.0,
            "tp1_close_fraction": 0.40,
            "tp2_close_fraction": 0.60,
            "cooldown_bars": 12,
            "max_bars_override": 48,
            "confidence": 0.72,
            "size_multiplier": 0.70,
        }
    if objective == "density":
        params["cooldown_bars"] = min(int(params.get("cooldown_bars", 14)), 12)
        params["min_adx"] = min(float(params.get("min_adx", 19.0)), 20.0)
        params["min_bb_rank"] = min(float(params.get("min_bb_rank", 0.26)), 0.30)
        params["min_atr_rank"] = min(float(params.get("min_atr_rank", 0.22)), 0.24)
    elif objective == "profit_factor":
        params["tp1_rr"] = max(float(params.get("tp1_rr", 2.0)), 2.1)
        params["tp2_rr"] = max(float(params.get("tp2_rr", 3.2)), 3.6)
    return params


def _mutate_trend_params(rng: random.Random, params: dict) -> dict:
    p = copy.deepcopy(params)
    p["entry_mode"] = "trend_pullback"
    p["use_htf_filter"] = True
    p["use_trend_filter"] = True
    p["use_volume_filter"] = True
    p["use_structure_filter"] = True
    p.setdefault("min_adx", 22.0)
    p.setdefault("htf_adx_min", 18.0)
    p.setdefault("htf_bb_rank_min", 0.30)
    p.setdefault("min_bb_rank", 0.38)
    p.setdefault("min_atr_rank", 0.32)
    p.setdefault("rsi_min", 52.0)
    p.setdefault("rsi_max", 70.0)
    p.setdefault("volume_multiplier", 1.08)
    p.setdefault("pullback_lookback", 3)
    p.setdefault("pullback_bars", 1)
    p.setdefault("stop_atr_mult", 2.0)
    p.setdefault("tp1_rr", 2.3)
    p.setdefault("tp2_rr", 4.0)
    p.setdefault("tp1_close_fraction", 0.50)
    p.setdefault("tp2_close_fraction", 0.50)
    p.setdefault("trail_atr_mult", 1.5)
    p.setdefault("cooldown_bars", 30)
    p.setdefault("max_bars_override", 84)
    p.setdefault("confidence", 0.84)
    p.setdefault("size_multiplier", 0.80)

    p["min_adx"] = _clamp(_safe_float(p.get("min_adx", 22.0), 22.0) + rng.uniform(-2.0, 3.0), 14.0, 30.0)
    p["min_bb_rank"] = _clamp(_safe_float(p.get("min_bb_rank", 0.38), 0.38) + rng.uniform(-0.05, 0.05), 0.10, 0.60)
    p["min_atr_rank"] = _clamp(_safe_float(p.get("min_atr_rank", 0.32), 0.32) + rng.uniform(-0.05, 0.05), 0.08, 0.50)
    p["htf_adx_min"] = _clamp(_safe_float(p.get("htf_adx_min", 18.0), 18.0) + rng.uniform(-2.0, 2.0), 10.0, 28.0)
    p["htf_bb_rank_min"] = _clamp(_safe_float(p.get("htf_bb_rank_min", 0.30), 0.30) + rng.uniform(-0.04, 0.04), 0.05, 0.50)
    p["rsi_min"] = _clamp(_safe_float(p.get("rsi_min", 52.0), 52.0) + rng.uniform(-3.0, 2.0), 40.0, 60.0)
    p["rsi_max"] = _clamp(_safe_float(p.get("rsi_max", 70.0), 70.0) + rng.uniform(-2.0, 3.0), 60.0, 80.0)
    p["volume_multiplier"] = _clamp(_safe_float(p.get("volume_multiplier", 1.08), 1.08) + rng.uniform(-0.08, 0.12), 0.90, 1.25)
    p["pullback_lookback"] = int(_clamp(int(_safe_float(p.get("pullback_lookback", 3), 3)) + rng.choice([-1, 0, 1]), 2, 6))
    p["pullback_bars"] = int(_clamp(int(_safe_float(p.get("pullback_bars", 1), 1)) + rng.choice([0, 1]), 1, 3))
    p["stop_atr_mult"] = _clamp(_safe_float(p.get("stop_atr_mult", 2.0), 2.0) + rng.uniform(-0.25, 0.35), 1.2, 3.0)
    p["tp1_rr"] = _clamp(_safe_float(p.get("tp1_rr", 2.3), 2.3) + rng.uniform(-0.35, 0.35), 1.4, 3.5)
    p["tp2_rr"] = _clamp(max(_safe_float(p.get("tp2_rr", 4.0), 4.0), p["tp1_rr"] + 0.8) + rng.uniform(-0.5, 0.6), 2.5, 6.5)
    p["tp1_close_fraction"] = _clamp(_safe_float(p.get("tp1_close_fraction", 0.50), 0.50) + rng.uniform(-0.10, 0.10), 0.20, 0.80)
    p["tp2_close_fraction"] = _clamp(1.0 - p["tp1_close_fraction"], 0.20, 0.80)
    p["trail_atr_mult"] = _clamp(_safe_float(p.get("trail_atr_mult", 1.5), 1.5) + rng.uniform(-0.20, 0.20), 1.0, 2.5)
    p["cooldown_bars"] = int(_clamp(int(_safe_float(p.get("cooldown_bars", 30), 30)) + rng.choice([-6, -3, 0, 3, 6]), 10, 40))
    p["max_bars_override"] = int(_clamp(int(_safe_float(p.get("max_bars_override", 84), 84)) + rng.choice([-18, -12, 0, 12, 18]), 30, 144))
    p["confidence"] = _clamp(_safe_float(p.get("confidence", 0.84), 0.84) + rng.uniform(-0.05, 0.05), 0.55, 0.95)
    p["size_multiplier"] = _clamp(_safe_float(p.get("size_multiplier", 0.80), 0.80) + rng.uniform(-0.15, 0.10), 0.35, 1.00)

    if rng.random() < 0.25:
        p["use_volume_filter"] = False
    if rng.random() < 0.20:
        p["use_structure_filter"] = False
    if rng.random() < 0.15:
        p["trail_ema20"] = True
    return p


def _mutate_breakout_params(rng: random.Random, params: dict) -> dict:
    p = copy.deepcopy(params)
    p["entry_mode"] = "breakout"
    p["use_htf_filter"] = True
    p["use_volume_filter"] = True
    p["use_breakout_filter"] = True
    p.setdefault("min_adx", 18.0)
    p.setdefault("min_bb_rank", 0.25)
    p.setdefault("stop_atr_mult", 1.7)
    p.setdefault("tp1_rr", 2.1)
    p.setdefault("tp2_rr", 3.5)
    p.setdefault("tp1_close_fraction", 0.40)
    p.setdefault("tp2_close_fraction", 0.60)
    p.setdefault("cooldown_bars", 24)
    p.setdefault("max_bars_override", 60)

    p["min_adx"] = _clamp(_safe_float(p.get("min_adx", 18.0), 18.0) + rng.uniform(-2.0, 4.0), 12.0, 28.0)
    p["min_bb_rank"] = _clamp(_safe_float(p.get("min_bb_rank", 0.25), 0.25) + rng.uniform(-0.05, 0.05), 0.08, 0.45)
    p["stop_atr_mult"] = _clamp(_safe_float(p.get("stop_atr_mult", 1.7), 1.7) + rng.uniform(-0.20, 0.30), 1.0, 2.6)
    p["tp1_rr"] = _clamp(_safe_float(p.get("tp1_rr", 2.1), 2.1) + rng.uniform(-0.25, 0.25), 1.4, 3.2)
    p["tp2_rr"] = _clamp(max(_safe_float(p.get("tp2_rr", 3.5), 3.5), p["tp1_rr"] + 0.8) + rng.uniform(-0.4, 0.5), 2.2, 5.5)
    p["tp1_close_fraction"] = _clamp(_safe_float(p.get("tp1_close_fraction", 0.40), 0.40) + rng.uniform(-0.08, 0.08), 0.20, 0.60)
    p["tp2_close_fraction"] = _clamp(1.0 - p["tp1_close_fraction"], 0.30, 0.80)
    p["cooldown_bars"] = int(_clamp(int(_safe_float(p.get("cooldown_bars", 24), 24)) + rng.choice([-6, -3, 0, 3, 6]), 8, 30))
    p["max_bars_override"] = int(_clamp(int(_safe_float(p.get("max_bars_override", 60), 60)) + rng.choice([-12, -6, 0, 6, 12]), 24, 96))
    p["confidence"] = _clamp(_safe_float(p.get("confidence", 0.78), 0.78) + rng.uniform(-0.05, 0.05), 0.50, 0.92)
    p["size_multiplier"] = _clamp(_safe_float(p.get("size_multiplier", 0.65), 0.65) + rng.uniform(-0.12, 0.10), 0.30, 0.90)
    return p


def _mutate_mean_reversion_params(rng: random.Random, params: dict) -> dict:
    p = copy.deepcopy(params)
    p["entry_mode"] = "mean_reversion"
    p["use_htf_filter"] = bool(p.get("use_htf_filter", True))
    p["use_reclaim_filter"] = True
    p["use_structure_filter"] = bool(p.get("use_structure_filter", False)) or rng.random() < 0.35
    p["use_volume_filter"] = bool(p.get("use_volume_filter", False))
    p["min_bb_rank"] = _clamp(_safe_float(p.get("min_bb_rank", 0.30), 0.30) + rng.uniform(-0.04, 0.03), 0.06, 0.40)
    p["rsi_max"] = _clamp(_safe_float(p.get("rsi_max", 32.0), 32.0) + rng.uniform(-4.0, 2.0), 18.0, 42.0)
    p["stop_atr_mult"] = _clamp(_safe_float(p.get("stop_atr_mult", 1.6), 1.6) + rng.uniform(-0.15, 0.25), 1.0, 2.2)
    p["tp1_rr"] = _clamp(_safe_float(p.get("tp1_rr", 1.8), 1.8) + rng.uniform(-0.15, 0.20), 1.2, 2.8)
    p["cooldown_bars"] = int(_clamp(int(_safe_float(p.get("cooldown_bars", 18), 18)) + rng.choice([-4, -2, 0, 2, 4]), 6, 22))
    p["max_bars_override"] = int(_clamp(int(_safe_float(p.get("max_bars_override", 36), 36)) + rng.choice([-6, 0, 6, 12]), 18, 60))
    p["confidence"] = _clamp(_safe_float(p.get("confidence", 0.65), 0.65) + rng.uniform(-0.04, 0.05), 0.45, 0.88)
    p["size_multiplier"] = _clamp(_safe_float(p.get("size_multiplier", 0.60), 0.60) + rng.uniform(-0.10, 0.08), 0.25, 0.85)
    p["tp2_rr"] = _clamp(_safe_float(p.get("tp2_rr", max(p["tp1_rr"] + 0.8, 2.8)), 2.8) + rng.uniform(-0.25, 0.35), p["tp1_rr"] + 0.5, 4.5)
    return p


def _mutate_mean_reversion_vwap_params(rng: random.Random, params: dict) -> dict:
    p = _mutate_mean_reversion_params(rng, params)
    p["entry_mode"] = "mean_reversion"
    p["use_volume_filter"] = True
    p["use_htf_filter"] = True
    p["use_structure_filter"] = True
    p["use_reclaim_filter"] = True
    p["min_bb_rank"] = _clamp(_safe_float(p.get("min_bb_rank", 0.24), 0.24) + rng.uniform(-0.03, 0.02), 0.05, 0.35)
    p["rsi_max"] = _clamp(_safe_float(p.get("rsi_max", 30.0), 30.0) + rng.uniform(-2.5, 1.5), 18.0, 38.0)
    p["volume_multiplier"] = _clamp(_safe_float(p.get("volume_multiplier", 1.02), 1.02) + rng.uniform(-0.08, 0.10), 0.85, 1.25)
    p["cooldown_bars"] = int(_clamp(int(_safe_float(p.get("cooldown_bars", 16), 16)) + rng.choice([-2, 0, 2, 4]), 5, 18))
    p["tp1_rr"] = _clamp(_safe_float(p.get("tp1_rr", 1.7), 1.7) + rng.uniform(-0.10, 0.15), 1.1, 2.4)
    p["tp2_rr"] = _clamp(max(_safe_float(p.get("tp2_rr", 2.6), 2.6), p["tp1_rr"] + 0.6) + rng.uniform(-0.20, 0.25), 2.0, 4.2)
    p["size_multiplier"] = _clamp(_safe_float(p.get("size_multiplier", 0.55), 0.55) + rng.uniform(-0.08, 0.06), 0.20, 0.75)
    return p


def _mutate_mean_reversion_extreme_params(rng: random.Random, params: dict) -> dict:
    p = _mutate_mean_reversion_params(rng, params)
    p["entry_mode"] = "mean_reversion"
    p["use_volume_filter"] = False
    p["use_htf_filter"] = True
    p["use_structure_filter"] = False
    p["use_reclaim_filter"] = True
    p["min_bb_rank"] = _clamp(_safe_float(p.get("min_bb_rank", 0.30), 0.30) + rng.uniform(0.00, 0.03), 0.08, 0.42)
    p["rsi_max"] = _clamp(_safe_float(p.get("rsi_max", 26.0), 26.0) + rng.uniform(-3.0, 1.0), 16.0, 35.0)
    p["stop_atr_mult"] = _clamp(_safe_float(p.get("stop_atr_mult", 1.55), 1.55) + rng.uniform(-0.20, 0.15), 1.0, 2.0)
    p["tp1_rr"] = _clamp(_safe_float(p.get("tp1_rr", 1.6), 1.6) + rng.uniform(-0.10, 0.20), 1.1, 2.2)
    p["tp2_rr"] = _clamp(max(_safe_float(p.get("tp2_rr", 2.4), 2.4), p["tp1_rr"] + 0.7) + rng.uniform(-0.15, 0.20), 2.0, 4.0)
    p["cooldown_bars"] = int(_clamp(int(_safe_float(p.get("cooldown_bars", 14), 14)) + rng.choice([-4, -2, 0, 2]), 4, 16))
    p["max_bars_override"] = int(_clamp(int(_safe_float(p.get("max_bars_override", 28), 28)) + rng.choice([-4, 0, 4, 8]), 14, 48))
    p["size_multiplier"] = _clamp(_safe_float(p.get("size_multiplier", 0.50), 0.50) + rng.uniform(-0.08, 0.05), 0.20, 0.70)
    return p


def _mutate_volatility_squeeze_breakout_params(rng: random.Random, params: dict) -> dict:
    p = _mutate_breakout_params(rng, params)
    p["entry_mode"] = "breakout"
    p["use_breakout_filter"] = True
    p["use_volume_filter"] = True
    p["use_htf_filter"] = True
    p["use_structure_filter"] = False
    p["min_bb_rank"] = _clamp(_safe_float(p.get("min_bb_rank", 0.22), 0.22) + rng.uniform(-0.03, 0.03), 0.08, 0.35)
    p["min_atr_rank"] = _clamp(_safe_float(p.get("min_atr_rank", 0.18), 0.18) + rng.uniform(-0.03, 0.03), 0.08, 0.35)
    p["volume_multiplier"] = _clamp(_safe_float(p.get("volume_multiplier", 1.0), 1.0) + rng.uniform(0.00, 0.12), 0.95, 1.35)
    p["cooldown_bars"] = int(_clamp(int(_safe_float(p.get("cooldown_bars", 20), 20)) + rng.choice([-4, -2, 0, 2]), 6, 26))
    p["tp1_rr"] = _clamp(_safe_float(p.get("tp1_rr", 2.1), 2.1) + rng.uniform(-0.15, 0.20), 1.4, 3.0)
    p["tp2_rr"] = _clamp(max(_safe_float(p.get("tp2_rr", 3.3), 3.3), p["tp1_rr"] + 0.7) + rng.uniform(-0.25, 0.35), 2.2, 5.2)
    return p


def _llm_mutations(base_params: dict, feedback: dict, n: int, llm_client) -> List[dict]:
    context = {
        "symbol": feedback.get("symbol"),
        "timeframe": feedback.get("timeframe"),
        "mutation_directives": feedback.get("mutation_directives"),
    }
    prompts = build_child_batch_prompts(context, n=n)
    jobs = [PromptJob(name=p["goal"], prompt=p["prompt"]) for p in prompts]
    results = batch_prompts_sync(jobs, client=llm_client, max_concurrency=min(4, n))

    out = []
    for _, text in results.items():
        try:
            data = json.loads(text) if isinstance(text, str) else text
            updates = data.get("parameter_updates") if isinstance(data, dict) else {}
            params = dict(base_params)
            if isinstance(updates, dict):
                params.update(updates)
            out.append(params)
        except Exception:
            out.append(dict(base_params))
    return out


def mutate_parent(parent, symbol, timeframe, n_children=4, seed=None, feedback=None, llm_client=None, diversity_pool=None):
    rng = random.Random(seed)
    base_params = dict((parent or {}).get("parameters") or {})

    if feedback is None:
        feedback = build_feedback_summary(symbol=symbol, timeframe=timeframe)

    objective = _objective_from_feedback(feedback)

    if llm_client is None:
        llm_client = get_default_llm_client()

    directives = (feedback or {}).get("mutation_directives") or {}
    base_entry_mode = str(base_params.get("entry_mode") or directives.get("entry_mode") or "trend_pullback")

    # Oversample more aggressively for mean-reversion so the search keeps a
    # larger survivor set after strict gates are applied.
    oversample = max(n_children * (5 if objective in {"density", "profit_factor"} else 3), n_children + 4)
    raw_params: List[dict] = []

    raw_params.append(_apply_safety_constraints(_baseline_params(symbol, objective), objective, symbol))

    # Add a small family of purpose-built templates so we can test a couple of
    # new mean-reversion and volatility archetypes alongside the existing pool.
    raw_params.append(_apply_safety_constraints(_mutate_mean_reversion_vwap_params(rng, _baseline_params(symbol, "profit_factor")), "profit_factor", symbol))
    raw_params.append(_apply_safety_constraints(_mutate_mean_reversion_extreme_params(rng, _baseline_params(symbol, "stability")), "stability", symbol))
    raw_params.append(_apply_safety_constraints(_mutate_volatility_squeeze_breakout_params(rng, _baseline_params(symbol, "density")), "density", symbol))

    if llm_client:
        llm_sets = _llm_mutations(base_params, feedback, oversample, llm_client)
        for llm_params in llm_sets:
            raw_params.append(_apply_safety_constraints(llm_params, objective, symbol))

    for _ in range(oversample):
        params = dict(base_params)
        params = _apply_directives(params, directives, symbol)

        if objective in {"density", "profit_factor"}:
            mode = rng.choices(
                ["mean_reversion", "breakout", "trend_pullback"],
                weights=[0.60, 0.25, 0.15],
                k=1,
            )[0]
        else:
            mode = base_entry_mode
            if rng.random() < 0.60:
                mode = rng.choice(["trend_pullback", "breakout", "mean_reversion"])

        if mode == "trend_pullback":
            params = _mutate_trend_params(rng, params)
        elif mode == "breakout":
            params = _mutate_volatility_squeeze_breakout_params(rng, params)
        else:
            if rng.random() < 0.55:
                params = _mutate_mean_reversion_vwap_params(rng, params)
            else:
                params = _mutate_mean_reversion_extreme_params(rng, params)

        raw_params.append(_apply_safety_constraints(params, objective, symbol))

    scored = []
    seen = set()

    for params in raw_params:
        sig = _signature(params)
        if sig in seen:
            continue

        if diversity_pool:
            too_close = False
            threshold = 0.07 if str(params.get("entry_mode") or "") == "mean_reversion" else 0.10
            for p in diversity_pool:
                if _distance(params, p.get("parameters", {})) < threshold:
                    too_close = True
                    break
            if too_close:
                continue

        priority = _candidate_priority(params, objective)
        priority += rng.uniform(0.0, 0.05)

        scored.append((priority, params))
        seen.add(sig)

    scored.sort(key=lambda x: x[0], reverse=True)
    selected = scored[: max(1, n_children)]

    candidates = []
    for _, params in selected:
        sid = f"evo_{symbol.replace('/', '_').lower()}_{timeframe}_{rng.randint(1,999999)}"
        archetype = str(params.get("entry_mode") or "mixed")
        if params.get("use_reclaim_filter") and params.get("use_volume_filter"):
            archetype = f"mr_vwap_{archetype}"
        elif params.get("use_reclaim_filter"):
            archetype = f"mr_extreme_{archetype}"
        elif params.get("use_breakout_filter"):
            archetype = f"vol_squeeze_{archetype}"

        candidates.append(
            StrategyCandidate(
                sid,
                str((parent or {}).get("strategy_id") or "seed"),
                int((parent or {}).get("version", 0) or 0) + 1,
                params,
                symbol,
                timeframe,
                [symbol, timeframe, "evo", archetype],
                "evolution",
                notes=f"objective={objective}; archetype={archetype}",
            )
        )

    return candidates


def seed_strategy(symbol, timeframe, family="evo"):
    return StrategyCandidate(
        f"{family}_{symbol.replace('/', '_').lower()}_{timeframe}_{random.randint(1,999999)}",
        "seed",
        1,
        {},
        symbol,
        timeframe,
        [symbol, timeframe, family],
        "seed",
    )
