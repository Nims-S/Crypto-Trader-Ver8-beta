from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from execution.router import select_active_strategy, route_strategies
from strategy.regime_classifier import classify_market


@dataclass(frozen=True)
class LiveRegimeSnapshot:
    symbol: str
    timeframe: str
    regime: str
    confidence: float
    features: dict[str, float]
    htf_features: dict[str, float] | None = None
    notes: str = ""


@dataclass(frozen=True)
class LiveRouteDecision:
    symbol: str
    timeframe: str
    regime: str
    strategy_id: str | None
    confidence: float
    routed_strategy: dict[str, Any] | None
    features: dict[str, float]
    htf_features: dict[str, float] | None = None


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _row_to_frame(row: dict[str, Any] | None) -> pd.DataFrame:
    if not row:
        return pd.DataFrame()
    return pd.DataFrame([row])


def _normalize_features(features: dict[str, Any] | None) -> dict[str, float]:
    features = features or {}
    out: dict[str, float] = {}
    for key in (
        "adx",
        "bb_width_rank",
        "bbwp_rank",
        "trend_strength",
        "volume_rank",
        "atr_rank",
        "liquidity_rank",
        "volatility_rank",
        "slope",
        "sma200",
        "close",
    ):
        if key in features:
            out[key] = _safe_float(features.get(key), 0.0)
    return out


def detect_live_regime(
    features: dict[str, Any] | None,
    *,
    htf_features: dict[str, Any] | None = None,
    symbol: str = "BTC/USDT",
    timeframe: str = "1d",
) -> LiveRegimeSnapshot:
    """Detect a coarse live market regime from feature snapshots.

    The detector is intentionally aligned with the existing regime classifier so
    live routing and research classification speak the same language.
    """
    features = features or {}
    htf_features = htf_features or {}

    df = _row_to_frame(features)
    htf_df = _row_to_frame(htf_features) if htf_features else None
    regime = classify_market(df, htf_df)

    feat = _normalize_features(features)
    htf_feat = _normalize_features(htf_features) if htf_features else None

    adx = feat.get("adx", 0.0)
    bb_rank = feat.get("bb_width_rank", feat.get("bbwp_rank", 0.0))
    volume_rank = feat.get("volume_rank", 0.0)
    atr_rank = feat.get("atr_rank", 0.0)
    slope = feat.get("slope", 0.0)
    liquidity_rank = feat.get("liquidity_rank", 0.0)
    vol_rank = feat.get("volatility_rank", 0.0)

    confidence = 0.35
    notes = []

    if regime == "trend":
        confidence += min(0.45, max(0.0, (adx - 18.0) / 20.0) * 0.25)
        confidence += min(0.20, max(0.0, bb_rank) * 0.10)
        if htf_feat and htf_feat.get("close", 0.0) > htf_feat.get("sma200", 0.0):
            confidence += 0.08
        notes.append("trend_bias")
    elif regime == "breakout":
        confidence += min(0.40, max(0.0, bb_rank - 0.45) * 0.50)
        confidence += min(0.15, max(0.0, volume_rank) * 0.12)
        confidence += min(0.10, max(0.0, vol_rank) * 0.10)
        notes.append("expansion_bias")
    elif regime == "mean_reversion":
        confidence += min(0.30, max(0.0, 0.45 - bb_rank) * 0.45)
        confidence += min(0.15, max(0.0, 1.0 - abs(slope)) * 0.12)
        confidence += min(0.10, max(0.0, 1.0 - atr_rank) * 0.08)
        notes.append("compression_bias")
    else:
        notes.append("fallback_regime")

    # Penalize weak liquidity / malformed feature snapshots.
    confidence -= min(0.15, max(0.0, 0.25 - liquidity_rank) * 0.20)
    confidence -= min(0.10, max(0.0, 0.15 - atr_rank) * 0.15)
    confidence = max(0.05, min(0.95, confidence))

    return LiveRegimeSnapshot(
        symbol=symbol,
        timeframe=timeframe,
        regime=regime,
        confidence=round(float(confidence), 6),
        features=feat,
        htf_features=htf_feat,
        notes=", ".join(notes),
    )


def route_live_strategy(
    symbol: str,
    timeframe: str,
    features: dict[str, Any],
    *,
    htf_features: dict[str, Any] | None = None,
    limit: int = 5,
) -> LiveRouteDecision:
    snapshot = detect_live_regime(features, htf_features=htf_features, symbol=symbol, timeframe=timeframe)
    strategy = select_active_strategy(symbol, timeframe, snapshot.regime, limit=limit)
    strategy_id = strategy.get("strategy_id") if strategy else None
    return LiveRouteDecision(
        symbol=symbol,
        timeframe=timeframe,
        regime=snapshot.regime,
        strategy_id=strategy_id,
        confidence=snapshot.confidence,
        routed_strategy=strategy,
        features=snapshot.features,
        htf_features=snapshot.htf_features,
    )


def route_live_strategies_from_snapshot(
    snapshot_map: dict[str, dict[str, Any]],
    *,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Route multiple symbol/timeframe snapshots.

    snapshot_map keys should be of the form `SYMBOL|TIMEFRAME`.
    Each value may include `features`, `htf_features`, `symbol`, and `timeframe`.
    """
    routed: list[dict[str, Any]] = []
    for key, payload in snapshot_map.items():
        symbol = str((payload or {}).get("symbol") or key.split("|")[0])
        timeframe = str((payload or {}).get("timeframe") or key.split("|")[1] if "|" in key else "1d")
        features = (payload or {}).get("features") or payload or {}
        htf_features = (payload or {}).get("htf_features")

        decision = route_live_strategy(symbol, timeframe, features, htf_features=htf_features, limit=limit)
        routed.append(
            {
                "symbol": decision.symbol,
                "timeframe": decision.timeframe,
                "regime": decision.regime,
                "confidence": decision.confidence,
                "strategy_id": decision.strategy_id,
                "strategy": decision.routed_strategy,
                "features": decision.features,
                "htf_features": decision.htf_features,
            }
        )
    return routed


def load_snapshot_file(path: str | Path) -> dict[str, dict[str, Any]]:
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("snapshot file must contain a JSON object")
    return data
