"""Validation helpers for the automated evolution loop."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
import pandas as pd

from research.scoring import score_metrics


@dataclass(frozen=True)
class WalkForwardSplit:
    label: str
    start: str
    end: str

    def as_dict(self) -> dict[str, str]:
        return {"label": self.label, "start": self.start, "end": self.end}


TRADE_DENSITY_BASE = {
    "1d": 4,
    "12h": 4,
    "8h": 5,
    "4h": 5,
    "2h": 6,
    "1h": 6,
    "30m": 8,
    "15m": 10,
}

SOFT_DENSITY_FLOOR = 0.30


def _to_utc_timestamp(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def _iso(ts: pd.Timestamp) -> str:
    return ts.tz_convert("UTC").isoformat() if ts.tzinfo else ts.tz_localize("UTC").isoformat()


def build_walk_forward_folds(start: str, end: str, *, folds: int = 3, train_ratio: float = 0.6, val_ratio: float = 0.2, test_ratio: float = 0.2) -> list[WalkForwardSplit]:
    start_ts = _to_utc_timestamp(start)
    end_ts = _to_utc_timestamp(end)

    span = end_ts - start_ts
    train_len = span * train_ratio
    val_len = span * val_ratio
    test_len = span * test_ratio
    step = test_len

    folds_out: list[WalkForwardSplit] = []
    for fold_idx in range(folds):
        fold_start = start_ts + (step * fold_idx)
        train_end = fold_start + train_len
        val_end = train_end + val_len
        test_end = val_end + test_len

        if test_end > end_ts:
            break

        folds_out.append(WalkForwardSplit(label=f"fold_{fold_idx + 1}", start=_iso(fold_start), end=_iso(test_end)))

    if not folds_out:
        folds_out.append(WalkForwardSplit(label="fold_1", start=_iso(start_ts), end=_iso(end_ts)))

    return folds_out


def _split_trade_floor(timeframe: str, split_name: str) -> int:
    base = TRADE_DENSITY_BASE.get((timeframe or "").lower(), 5)
    # much lower floors for splits
    if split_name == "train":
        return max(3, int(round(base * 0.5)))
    return max(2, int(round(base * 0.3)))


def summarize_walk_forward_reports(fold_reports: list[dict[str, Any]], *, timeframe: str) -> dict[str, Any]:
    if not fold_reports:
        return {"score": 0.0, "passed": False, "reason": "no fold reports", "fold_count": 0}

    split_scores = {"train": [], "val": [], "test": []}
    pass_counts = {"train": 0, "val": 0, "test": 0}
    total_counts = {"train": 0, "val": 0, "test": 0}

    for fold in fold_reports:
        for split in ("train", "val", "test"):
            result = fold.get(split) or {}
            decision = score_metrics(result, timeframe=timeframe, min_trades=_split_trade_floor(timeframe, split))
            split_scores[split].append(decision.score)
            total_counts[split] += 1
            if decision.passed:
                pass_counts[split] += 1

    train_mean = float(np.mean(split_scores["train"]))
    val_mean = float(np.mean(split_scores["val"]))
    test_mean = float(np.mean(split_scores["test"]))

    combined = split_scores["train"] + split_scores["val"] + split_scores["test"]
    score_spread = max(combined) - min(combined) if len(combined) > 1 else 0.0

    final_score = 0.2 * train_mean + 0.4 * val_mean + 0.4 * test_mean

    val_pass_ratio = pass_counts["val"] / max(1, total_counts["val"])
    test_pass_ratio = pass_counts["test"] / max(1, total_counts["test"])

    passed = (
        val_mean >= 0.25
        and test_mean >= 0.25
        and final_score >= 0.28
        and score_spread <= 1.0
        and val_pass_ratio >= 0.25
        and test_pass_ratio >= 0.25
    )

    reasons = []
    if val_mean < 0.25:
        reasons.append("val_weak")
    if test_mean < 0.25:
        reasons.append("test_weak")
    if score_spread > 1.0:
        reasons.append("wf_spread>1.0")

    return {
        "score": round(final_score, 6),
        "passed": passed,
        "reasons": reasons,
        "fold_count": len(fold_reports),
        "score_spread": round(score_spread, 6),
        "means": {"train": round(train_mean, 6), "val": round(val_mean, 6), "test": round(test_mean, 6)},
    }
