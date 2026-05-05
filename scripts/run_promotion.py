from __future__ import annotations

import argparse
import json

from research.promotion import promote_winners


def main() -> int:
    parser = argparse.ArgumentParser(description="Promote validated strategies")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--min-score", type=float, default=0.55)
    parser.add_argument("--commit", action="store_true", help="Write promotions back to registry")
    args = parser.parse_args()

    result = promote_winners(limit=args.limit, dry_run=not args.commit, min_score=args.min_score)
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
