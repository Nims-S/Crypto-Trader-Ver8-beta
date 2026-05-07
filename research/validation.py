"""Validation helpers for the automated evolution loop.

Walk-forward v2 uses anchored expanding training windows plus fixed validation
and test slices. This keeps the evaluation honest while avoiding the
pathological "everything is a tiny fold" behavior that can flatten scores.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from research.scoring import score_metrics


@dataclass(frozen=True)
class WalkForwardSplit:
    label: str
    start: str
    end: str
    train_start: str | None = None
    train_end: str | None = None
    val_start: str | None = None
    val_end: str | None = None
    test_start: str | None = None
    test_end: str | None = None

    def as_dict(self) -> dict[str, str]:
        payload: dict[str, str] = {"label": self.label, "start": self.start, "end": self.end}
        for key in ("train_start", "train_end", "val_start", "val_end", "test_start", "test_end"):
            value = getattr(self, key)
            if value is not None:
                payload[key] = value
        return payload


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

# Slightly stricter floor than backtest scoring because walk-forward should
# represent repeatability across unseen windows.
WALK_FORWARD_MIN_SCORE = 0.30
WALK_FORWARD_MIN_FINAL_SCORE = 0.32
WALK_FORWARD_MAX_SPREAD = 0.60
WALK_FORWARD_MIN_PASS_RATIO = 0.34


def _to_utc_timestamp(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def _iso(ts: pd.Timestamp) -> str:
    return ts.tz_convert("UTC").isoformat() if ts.tzinfo else ts.tz_localize("UTC").isoformat()


def _normalize_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> tuple[float, float, float]:
    total = float(train_ratio + val_ratio + test_ratio)
    if total <= 0:
        return 0.6, 0.2, 0.2
    return train_ratio / total, val_ratio / total, test_ratio / total


def _choose_window_lengths(span: pd.Timedelta, folds: int) -> tuple[pd.Timedelta, pd.Timedelta, pd.Timedelta]:
    """Pick practical walk-forward lengths from the total backtest span."""
    requested_folds = max(1, int(folds))
    if span <= pd.Timedelta(days=60):
        # Very short spans do not support meaningful walk-forward splits.
        return span * 0.60, span * 0.20, span * 0.20

    # Anchored expanding training window with fixed validation/test slices.
    test_len = max(pd.Timedelta(days=30), span * 0.12)
    val_len = max(pd.Timedelta(days=21), span * 0.08)
    initial_train_len = max(pd.Timedelta(days=120), span * 0.50)

    # If the caller asks for many folds, compress the test step but keep enough
    # room for a real validation/test pair.
    if requested_folds > 1:
        max_step = (span - initial_train_len - val_len - test_len) / max(1, requested_folds - 1)
        if max_step > pd.Timedelta(0):
            test_len = min(test_len, max_step)
            test_len = max(pd.Timedelta(days=21), test_len)

    return initial_train_len, val_len, test_len


def build_walk_forward_folds(
    start: str,
    end: str,
    *,
    folds: int = 3,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
) -> list[WalkForwardSplit]:
    """Build rolling evaluation windows for walk-forward validation.

    Walk-forward v2 uses anchored expanding windows. The fold boundaries are
    chosen once, then each fold is split into train/validation/test by
    `split_walk_forward_window()`.
    """
    start_ts = _to_utc_timestamp(start)
    end_ts = _to_utc_timestamp(end)
    if end_ts <= start_ts:
        return [WalkForwardSplit(label="fold_1", start=_iso(start_ts), end=_iso(end_ts))]

    train_ratio, val_ratio, test_ratio = _normalize_ratios(train_ratio, val_ratio, test_ratio)
    span = end_ts - start_ts
    requested_folds = max(1, int(folds))

    initial_train_len, val_len, test_len = _choose_window_lengths(span, requested_folds)
    min_total_needed = initial_train_len + val_len + test_len

    if requested_folds == 1 or span <= min_total_needed + pd.Timedelta(days=7):
        # Fallback to a single anchored fold when the span is too short.
        train_end = start_ts + max(initial_train_len, span * train_ratio)
        val_end = min(end_ts, train_end + max(val_len, span * val_ratio))
        test_end = min(end_ts, val_end + max(test_len, span * test_ratio))
        return [
            WalkForwardSplit(
                label="fold_1",
                start=_iso(start_ts),
                end=_iso(test_end),
                train_start=_iso(start_ts),
                train_end=_iso(train_end),
                val_start=_iso(train_end),
                val_end=_iso(val_end),
                test_start=_iso(val_end),
                test_end=_iso(test_end),
            )
        ]

    step = test_len
    max_possible = int((span - min_total_needed) // step) + 1
    fold_count = max(1, min(requested_folds, max_possible))

    folds_out: list[WalkForwardSplit] = []
    for fold_idx in range(fold_count):
        train_end = start_ts + initial_train_len + (step * fold_idx)
        val_start = train_end
        val_end = val_start + val_len
        test_start = val_end
        test_end = test_start + test_len
        if test_end > end_ts:
            break
        if test_end <= start_ts:
            continue

        folds_out.append(
            WalkForwardSplit(
                label=f"fold_{fold_idx + 1}",
                start=_iso(start_ts),
                end=_iso(test_end),
                train_start=_iso(start_ts),
                train_end=_iso(train_end),
                val_start=_iso(val_start),
                val_end=_iso(val_end),
                test_start=_iso(test_start),
                test_end=_iso(test_end),
            )
        )

    if not folds_out:
        train_end = start_ts + max(initial_train_len, span * train_ratio)
        val_end = min(end_ts, train_end + max(val_len, span * val_ratio))
        test_end = min(end_ts, val_end + max(test_len, span * test_ratio))
        folds_out.append(
            WalkForwardSplit(
                label="fold_1",
                start=_iso(start_ts),
                end=_iso(test_end),
                train_start=_iso(start_ts),
                train_end=_iso(train_end),
                val_start=_iso(train_end),
                val_end=_iso(val_end),
                test_start=_iso(val_end),
                test_end=_iso(test_end),
            )
        )

    return folds_out


def split_walk_forward_window(
    fold: WalkForwardSplit,
    *,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
) -> dict[str, dict[str, str]]:
    """Split a fold window into train/val/test sections.

    If the fold already contains explicit boundaries from walk-forward v2, those
    are used directly. The ratio-based fallback preserves compatibility with
    older fold objects and external callers.
    """
    if fold.train_start and fold.train_end and fold.val_start and fold.val_end and fold.test_start and fold.test_end:
        return {
            "train": {"start": fold.train_start, "end": fold.train_end},
            "val": {"start": fold.val_start, "end": fold.val_end},
            "test": {"start": fold.test_start, "end": fold.test_end},
        }

    fold_start = _to_utc_timestamp(fold.start)
    fold_end = _to_utc_timestamp(fold.end)
    if fold_end <= fold_start:
        iso = _iso(fold_start)
        return {
            "train": {"start": iso, "end": iso},
            "val": {"start": iso, "end": iso},
            "test": {"start": iso, "end": iso},
        }

    train_ratio, val_ratio, test_ratio = _normalize_ratios(train_ratio, val_ratio, test_ratio)
    span = fold_end - fold_start

    train_end = fold_start + (span * train_ratio)
    val_end = train_end + (span * val_ratio)
    test_end = fold_end

    # Avoid zero-width sections in short windows.
    min_slice = pd.Timedelta(days=3)
    if train_end - fold_start < min_slice:
        train_end = fold_start + min_slice
    if val_end - train_end < min_slice:
        val_end = train_end + min_slice
    if test_end - val_end < min_slice:
        val_end = max(train_end + min_slice, fold_end - min_slice)

    if val_end > fold_end:
        val_end = fold_end
    if train_end > val_end:
        train_end = val_end

    return {
        "train": {"start": _iso(fold_start), "end": _iso(train_end)},
        "val": {"start": _iso(train_end), "end": _iso(val_end)},
        "test": {"start": _iso(val_end), "end": _iso(test_end)},
    }


def _split_trade_floor(timeframe: str, split_name: str) -> int:
    base = TRADE_DENSITY_BASE.get((timeframe or "").lower(), 5)
    if split_name == "train":
        return max(3, int(round(base * 0.5)))
    if split_name == "val":
        return max(2, int(round(base * 0.35)))
    return max(2, int(round(base * 0.35)))


def summarize_walk_forward_reports(fold_reports: list[dict[str, Any]], *, timeframe: str) -> dict[str, Any]:
    if not fold_reports:
        return {"score": 0.0, "passed": False, "reason": "no fold reports", "fold_count": 0}

    split_scores = {"train": [], "val": [], "test": []}
    pass_counts = {"train": 0, "val": 0, "test": 0}
    total_counts = {"train": 0, "val": 0, "test": 0}
    trade_counts = {"train": [], "val": [], "test": []}
    pf_counts = {"train": [], "val": [], "test": []}
    wr_counts = {"train": [], "val": [], "test": []}
    dd_counts = {"train": [], "val": [], "test": []}

    for fold in fold_reports:
        for split in ("train", "val", "test"):
            result = fold.get(split) or {}
            floor = _split_trade_floor(timeframe, split)
            decision = score_metrics(result, timeframe=timeframe, min_trades=floor)
            split_scores[split].append(decision.score)
            total_counts[split] += 1
            if decision.passed:
                pass_counts[split] += 1

            trade_counts[split].append(float(result.get("trades", 0) or 0))
            pf_counts[split].append(float(result.get("profit_factor", 0) or 0))
            wr_counts[split].append(float(result.get("win_rate", 0) or 0))
            dd_counts[split].append(abs(float(result.get("max_drawdown_pct", 0) or 0)))

    train_mean = float(np.mean(split_scores["train"]))
    val_mean = float(np.mean(split_scores["val"]))
    test_mean = float(np.mean(split_scores["test"]))

    combined = split_scores["train"] + split_scores["val"] + split_scores["test"]
    score_spread = max(combined) - min(combined) if len(combined) > 1 else 0.0

    final_score = 0.2 * train_mean + 0.4 * val_mean + 0.4 * test_mean

    val_pass_ratio = pass_counts["val"] / max(1, total_counts["val"])
    test_pass_ratio = pass_counts["test"] / max(1, total_counts["test"])

    mean_trades = {k: float(np.mean(v)) if v else 0.0 for k, v in trade_counts.items()}
    mean_pf = {k: float(np.mean(v)) if v else 0.0 for k, v in pf_counts.items()}
    mean_wr = {k: float(np.mean(v)) if v else 0.0 for k, v in wr_counts.items()}
    mean_dd = {k: float(np.mean(v)) if v else 0.0 for k, v in dd_counts.items()}

    passed = (
        val_mean >= WALK_FORWARD_MIN_SCORE
        and test_mean >= WALK_FORWARD_MIN_SCORE
        and final_score >= WALK_FORWARD_MIN_FINAL_SCORE
        and score_spread <= WALK_FORWARD_MAX_SPREAD
        and val_pass_ratio >= WALK_FORWARD_MIN_PASS_RATIO
        and test_pass_ratio >= WALK_FORWARD_MIN_PASS_RATIO
    )

    reasons = []
    if val_mean < WALK_FORWARD_MIN_SCORE:
        reasons.append("val_weak")
    if test_mean < WALK_FORWARD_MIN_SCORE:
        reasons.append("test_weak")
    if score_spread > WALK_FORWARD_MAX_SPREAD:
        reasons.append("wf_spread_high")
    if val_pass_ratio < WALK_FORWARD_MIN_PASS_RATIO:
        reasons.append("val_pass_ratio_low")
    if test_pass_ratio < WALK_FORWARD_MIN_PASS_RATIO:
        reasons.append("test_pass_ratio_low")

    return {
        "score": round(final_score, 6),
        "passed": passed,
        "reasons": reasons,
        "fold_count": len(fold_reports),
        "score_spread": round(score_spread, 6),
        "means": {"train": round(train_mean, 6), "val": round(val_mean, 6), "test": round(test_mean, 6)},
        "trade_counts": {k: round(v, 6) for k, v in mean_trades.items()},
        "profit_factor": {k: round(v, 6) for k, v in mean_pf.items()},
        "win_rate": {k: round(v, 6) for k, v in mean_wr.items()},
        "max_drawdown": {k: round(v, 6) for k, v in mean_dd.items()},
        "pass_ratios": {"train": round(pass_counts["train"] / max(1, total_counts["train"]), 6), "val": round(val_pass_ratio, 6), "test": round(test_pass_ratio, 6)},
    }
