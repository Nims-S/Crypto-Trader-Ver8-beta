from __future__ import annotations

from typing import Dict, Any


def update_runtime(runtime: Dict[str, Any], *, live: dict[str, Any], drift: dict[str, Any], cycle: int) -> Dict[str, Any]:
    r = dict(runtime or {})
    r["cycles_seen"] = int(r.get("cycles_seen", 0)) + 1
    r["last_cycle"] = int(cycle)
    if "first_cycle" not in r:
        r["first_cycle"] = int(cycle)

    last_status = drift.get("status")
    r["last_status"] = last_status

    # simple cooldown if disabled
    if last_status == "disable":
        r["cooldown_until"] = cycle + 5
    else:
        r["cooldown_until"] = int(r.get("cooldown_until", 0))

    # track simple streaks
    pnl = float(live.get("pnl", 0.0) or 0.0)
    if pnl < 0:
        r["loss_streak"] = int(r.get("loss_streak", 0)) + 1
        r["win_streak"] = 0
    else:
        r["win_streak"] = int(r.get("win_streak", 0)) + 1
        r["loss_streak"] = 0

    return r


def lifecycle_multiplier(runtime: Dict[str, Any], cycle: int) -> float:
    if not runtime:
        return 0.5  # warmup

    cooldown_until = int(runtime.get("cooldown_until", 0))
    if cooldown_until and cycle < cooldown_until:
        return 0.0

    cycles = int(runtime.get("cycles_seen", 0))
    if cycles < 3:
        return 0.5

    loss_streak = int(runtime.get("loss_streak", 0))
    if loss_streak >= 3:
        return 0.3

    return 1.0
