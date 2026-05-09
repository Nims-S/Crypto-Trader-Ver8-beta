from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SeedParent:
    strategy_id: str
    base_strategy: str
    version: int
    parameters: dict[str, Any]
    symbol: str
    timeframe: str
    tags: list[str]
    source: str = "seed"
    notes: str = ""
    status: str = "candidate"
    active: bool = False
    regime_profile: str | None = None
    robustness_score: float = 0.0


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _slug(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in value).strip("_") or "unknown"


def _seed_id(symbol: str, timeframe: str, regime: str, archetype: str, params: dict[str, Any], index: int) -> str:
    blob = json.dumps([symbol, timeframe, regime, archetype, params, index], sort_keys=True, default=str).encode("utf-8")
    return f"seed_{_slug(symbol)}_{_slug(timeframe)}_{_slug(regime)}_{_slug(archetype)}_{hashlib.sha1(blob).hexdigest()[:10]}"


def _bias_for_symbol(symbol: str) -> dict[str, float]:
    if symbol.startswith("BTC"):
        return {"adx": 1.5, "bb": 0.02, "atr": 0.02, "rsi": 1.0, "volume": 0.02, "stop": -0.05}
    if symbol.startswith("ETH"):
        return {"adx": 0.5, "bb": -0.01, "atr": 0.04, "rsi": -0.5, "volume": 0.05, "stop": 0.03}
    if symbol.startswith("SOL"):
        return {"adx": -0.5, "bb": 0.00, "atr": 0.06, "rsi": -1.0, "volume": 0.08, "stop": 0.08}
    return {"adx": 0.0, "bb": 0.0, "atr": 0.0, "rsi": 0.0, "volume": 0.0, "stop": 0.0}


def _template(archetype: str, regime: str, symbol: str) -> dict[str, Any]:
    base = {
        "entry_mode": "mean_reversion" if archetype.startswith("mr_") else ("breakout" if "breakout" in archetype else "trend_pullback"),
        "use_htf_filter": True,
        "use_volume_filter": False,
        "use_structure_filter": False,
        "use_reclaim_filter": False,
        "use_trend_filter": False,
        "use_breakout_filter": False,
        "confidence": 0.70,
        "size_multiplier": 0.65,
        "cooldown_bars": 14,
        "max_bars_override": 36,
        "stop_atr_mult": 1.7,
        "tp1_rr": 1.8,
        "tp2_rr": 2.8,
        "tp1_close_fraction": 0.45,
        "tp2_close_fraction": 0.55,
        "trail_atr_mult": 1.2,
        "min_adx": 18.0,
        "min_bb_rank": 0.22,
        "min_atr_rank": 0.18,
        "htf_adx_min": 16.0,
        "htf_bb_rank_min": 0.20,
        "rsi_min": 40.0,
        "rsi_max": 65.0,
        "volume_multiplier": 1.00,
    }

    if archetype == "trend_pullback_core":
        base.update(
            {
                "entry_mode": "trend_pullback",
                "use_volume_filter": True,
                "use_structure_filter": True,
                "use_trend_filter": True,
                "confidence": 0.82,
                "size_multiplier": 0.80,
                "cooldown_bars": 16,
                "max_bars_override": 84,
                "stop_atr_mult": 2.0,
                "tp1_rr": 2.2,
                "tp2_rr": 3.8,
                "tp1_close_fraction": 0.50,
                "tp2_close_fraction": 0.50,
                "min_adx": 20.0,
                "min_bb_rank": 0.30,
                "min_atr_rank": 0.24,
                "htf_adx_min": 18.0,
                "htf_bb_rank_min": 0.24,
                "rsi_min": 48.0,
                "rsi_max": 70.0,
                "volume_multiplier": 1.05,
            }
        )
    elif archetype == "vol_squeeze_breakout":
        base.update(
            {
                "entry_mode": "breakout",
                "use_volume_filter": True,
                "use_breakout_filter": True,
                "confidence": 0.76,
                "size_multiplier": 0.72,
                "cooldown_bars": 10,
                "max_bars_override": 48,
                "stop_atr_mult": 1.6,
                "tp1_rr": 1.9,
                "tp2_rr": 3.1,
                "min_adx": 17.0,
                "min_bb_rank": 0.18,
                "min_atr_rank": 0.15,
                "htf_adx_min": 15.0,
                "htf_bb_rank_min": 0.18,
                "volume_multiplier": 1.06,
                "tp1_close_fraction": 0.40,
                "tp2_close_fraction": 0.60,
            }
        )
    elif archetype == "mr_vwap_reclaim":
        base.update(
            {
                "entry_mode": "mean_reversion",
                "use_volume_filter": True,
                "use_structure_filter": True,
                "use_reclaim_filter": True,
                "confidence": 0.74,
                "size_multiplier": 0.60,
                "cooldown_bars": 10,
                "max_bars_override": 24,
                "stop_atr_mult": 1.45,
                "tp1_rr": 1.55,
                "tp2_rr": 2.35,
                "min_adx": 15.0,
                "min_bb_rank": 0.16,
                "min_atr_rank": 0.14,
                "htf_adx_min": 14.0,
                "htf_bb_rank_min": 0.16,
                "rsi_min": 42.0,
                "rsi_max": 34.0,
                "volume_multiplier": 1.02,
                "tp1_close_fraction": 0.52,
                "tp2_close_fraction": 0.48,
            }
        )
    elif archetype == "mr_extreme_fade":
        base.update(
            {
                "entry_mode": "mean_reversion",
                "use_reclaim_filter": True,
                "use_structure_filter": False,
                "use_volume_filter": False,
                "confidence": 0.70,
                "size_multiplier": 0.55,
                "cooldown_bars": 8,
                "max_bars_override": 20,
                "stop_atr_mult": 1.35,
                "tp1_rr": 1.40,
                "tp2_rr": 2.10,
                "min_adx": 14.0,
                "min_bb_rank": 0.20,
                "min_atr_rank": 0.12,
                "htf_adx_min": 12.0,
                "htf_bb_rank_min": 0.14,
                "rsi_min": 38.0,
                "rsi_max": 28.0,
                "volume_multiplier": 0.98,
                "tp1_close_fraction": 0.60,
                "tp2_close_fraction": 0.40,
            }
        )
    elif archetype == "mr_compression_revert":
        base.update(
            {
                "entry_mode": "mean_reversion",
                "use_volume_filter": True,
                "use_structure_filter": True,
                "use_reclaim_filter": True,
                "confidence": 0.72,
                "size_multiplier": 0.58,
                "cooldown_bars": 12,
                "max_bars_override": 28,
                "stop_atr_mult": 1.50,
                "tp1_rr": 1.60,
                "tp2_rr": 2.55,
                "min_adx": 15.5,
                "min_bb_rank": 0.18,
                "min_atr_rank": 0.16,
                "htf_adx_min": 13.0,
                "htf_bb_rank_min": 0.16,
                "rsi_min": 40.0,
                "rsi_max": 32.0,
                "volume_multiplier": 1.03,
                "tp1_close_fraction": 0.55,
                "tp2_close_fraction": 0.45,
            }
        )
    return base


def _apply_objective_bias(params: dict[str, Any], objective: str, regime: str, symbol: str, rng: random.Random) -> dict[str, Any]:
    p = dict(params)
    bias = _bias_for_symbol(symbol)
    mr = p.get("entry_mode") == "mean_reversion"

    if mr:
        p["min_bb_rank"] = _clamp(_safe_float(p.get("min_bb_rank", 0.20), 0.20) + bias["bb"], 0.05, 0.40)
        p["rsi_max"] = _clamp(_safe_float(p.get("rsi_max", 32.0), 32.0) + bias["rsi"], 18.0, 42.0)
        p["stop_atr_mult"] = _clamp(_safe_float(p.get("stop_atr_mult", 1.5), 1.5) + bias["stop"], 1.0, 2.2)
        p["min_atr_rank"] = _clamp(_safe_float(p.get("min_atr_rank", 0.16), 0.16) + bias["atr"], 0.08, 0.35)
        p["min_adx"] = _clamp(_safe_float(p.get("min_adx", 15.0), 15.0) + bias["adx"], 10.0, 24.0)
        if symbol.startswith("SOL"):
            p["max_bars_override"] = min(int(_safe_float(p.get("max_bars_override", 24), 24)), 24)
            p["size_multiplier"] = _clamp(_safe_float(p.get("size_multiplier", 0.55), 0.55), 0.35, 0.70)
        elif symbol.startswith("ETH"):
            p["volume_multiplier"] = _clamp(_safe_float(p.get("volume_multiplier", 1.00), 1.00) + 0.05, 0.85, 1.20)
        else:
            p["use_structure_filter"] = True

    if objective == "density":
        p["cooldown_bars"] = max(5, int(_safe_float(p.get("cooldown_bars", 12), 12)) - rng.choice([0, 1, 2]))
        p["max_bars_override"] = max(12, int(_safe_float(p.get("max_bars_override", 24), 24)) - rng.choice([0, 2, 4]))
    elif objective == "profit_factor":
        p["tp1_rr"] = _clamp(_safe_float(p.get("tp1_rr", 1.6), 1.6) + rng.uniform(0.0, 0.20), 1.1, 3.0)
        p["tp2_rr"] = _clamp(_safe_float(p.get("tp2_rr", 2.5), 2.5) + rng.uniform(0.0, 0.30), 1.8, 4.5)
    elif objective == "stability":
        p["confidence"] = _clamp(_safe_float(p.get("confidence", 0.70), 0.70) + rng.uniform(0.02, 0.08), 0.45, 0.95)
        p["size_multiplier"] = _clamp(_safe_float(p.get("size_multiplier", 0.60), 0.60) * 0.92, 0.25, 1.0)

    return p


def _archetypes_for_regime(regime: str) -> list[str]:
    regime = (regime or "trend").strip().lower()
    if regime == "mean_reversion":
        return ["mr_vwap_reclaim", "mr_extreme_fade", "mr_compression_revert", "vol_squeeze_breakout"]
    if regime == "breakout":
        return ["vol_squeeze_breakout", "mr_vwap_reclaim", "mr_extreme_fade"]
    return ["trend_pullback_core", "mr_vwap_reclaim", "vol_squeeze_breakout"]


def build_survivor_seed_parents(
    *,
    symbol: str,
    timeframe: str,
    regime: str,
    objective: str,
    count: int = 3,
    seed: int | None = None,
) -> list[dict[str, Any]]:
    """Create explicit seed parents to expand survivor generation.

    The search is intentionally biased toward mean-reversion, but it also keeps a
    volatility-compression breakout branch alive so the research loop can test a
    couple of orthogonal families in the same cycle.
    """
    rng = random.Random(seed if seed is not None else int(hashlib.sha1(f"{symbol}|{timeframe}|{regime}|{objective}".encode()).hexdigest()[:8], 16))
    archetypes = _archetypes_for_regime(regime)
    count = max(1, int(count))

    parents: list[dict[str, Any]] = []
    for idx in range(count):
        archetype = archetypes[idx % len(archetypes)]
        params = _template(archetype, regime, symbol)
        params = _apply_objective_bias(params, objective, regime, symbol, rng)

        # Gentle parameter diversification around the archetype, while keeping the
        # family recognizable for mutation and registry lineage tracking.
        if params.get("entry_mode") == "mean_reversion":
            params["tp1_rr"] = _clamp(_safe_float(params.get("tp1_rr", 1.6), 1.6) + rng.uniform(-0.12, 0.12), 1.1, 3.0)
            params["tp2_rr"] = _clamp(max(_safe_float(params.get("tp2_rr", 2.5), 2.5), params["tp1_rr"] + 0.6) + rng.uniform(-0.15, 0.20), 1.8, 4.8)
        else:
            params["tp1_rr"] = _clamp(_safe_float(params.get("tp1_rr", 1.9), 1.9) + rng.uniform(-0.15, 0.15), 1.2, 3.8)
            params["tp2_rr"] = _clamp(max(_safe_float(params.get("tp2_rr", 3.1), 3.1), params["tp1_rr"] + 0.7) + rng.uniform(-0.20, 0.25), 2.0, 6.0)

        if symbol.startswith("ETH") and archetype.startswith("mr_"):
            params["use_volume_filter"] = True
            params["volume_multiplier"] = _clamp(_safe_float(params.get("volume_multiplier", 1.02), 1.02) + 0.04, 0.85, 1.25)
        if symbol.startswith("SOL") and archetype.startswith("mr_"):
            params["min_bb_rank"] = _clamp(_safe_float(params.get("min_bb_rank", 0.18), 0.18) - 0.02, 0.05, 0.35)
            params["max_bars_override"] = min(int(_safe_float(params.get("max_bars_override", 24), 24)), 24)

        strategy_id = _seed_id(symbol, timeframe, regime, archetype, params, idx)
        parents.append(
            {
                "strategy_id": strategy_id,
                "base_strategy": f"seed_{archetype}",
                "version": 1,
                "parameters": params,
                "symbol": symbol,
                "timeframe": timeframe,
                "tags": [symbol, timeframe, regime, archetype, "seed"],
                "source": "seed",
                "notes": f"{objective}:{archetype}",
                "status": "candidate",
                "active": False,
                "regime_profile": regime,
                "robustness_score": 0.0,
            }
        )

    return parents
