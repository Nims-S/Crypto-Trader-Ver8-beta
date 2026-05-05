from __future__ import annotations

import argparse
import json

from config.defaults import DEFAULT_SYMBOLS, DEFAULT_TIMEFRAMES
from research.coordinator import evolve


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one research evolution cycle")
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--children-per-parent", type=int, default=4)
    parser.add_argument("--lookback-days", type=int, default=720)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    result = evolve(
        DEFAULT_SYMBOLS,
        DEFAULT_TIMEFRAMES,
        max_cycles=args.cycles,
        children_per_parent=args.children_per_parent,
        lookback_days=args.lookback_days,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
