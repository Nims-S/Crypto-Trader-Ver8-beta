from __future__ import annotations

import argparse
import json

from research.monte_carlo import run_monte_carlo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", default=None)
    parser.add_argument("--sims", type=int, default=1000)
    parser.add_argument("--horizon", type=int, default=None)
    args = parser.parse_args()

    result = run_monte_carlo(
        strategy_id=args.strategy,
        simulations=args.sims,
        horizon=args.horizon,
    )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
