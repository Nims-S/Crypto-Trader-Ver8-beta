import argparse

from execution.live_bot import run_live_cycle, run_loop


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--interval", type=int, default=None)
    parser.add_argument("--capital", type=float, default=None)
    args = parser.parse_args()

    if args.once:
        print(run_live_cycle(total_capital=args.capital))
    else:
        run_loop(interval_seconds=args.interval, total_capital=args.capital)
