# Crypto-Trader-Ver8-alpha

Ver8 reorganizes the bot into three layers:

- `research/` — offline evolution, validation, promotion
- `registry/` — strategy memory and lineage
- `execution/` — live routing, allocation, drift control
- `strategy/` — signal logic, indicators, regime detection

This scaffold is a migration target from the Ver7 branch.

## Quick Start

Install dependencies:

```
pip install -r requirements.txt
```

Run sanity check:

```
python scripts/sanity_check.py
```

Run backtest:

```
python main.py backtest --symbol BTC/USDT --timeframe 1d
```

View registry:

```
python main.py status
```

Rank strategies:

```
python main.py rank --symbol BTC/USDT --timeframe 1d
```
