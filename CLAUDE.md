# Quantitative Portfolio Optimizer

Markowitz mean-variance portfolio optimization. Fetches price data, estimates
expected returns and covariance, optimizes long-only portfolios (minimum
variance, maximum Sharpe / tangency, and a 1/N benchmark), renders Plotly
charts, and logs every run.

## Project layout
- `src/data/` — price fetching (yfinance) and parquet caching
- `src/models/` — expected returns, covariance estimators, optimizers, metrics
- `src/visualization/` — Plotly charts (efficient frontier, weights, correlation, performance)
- `src/utils/` — run history logging, CLAUDE.md sync, git sync, CLI parser
- `tests/` — pytest suite
- `main.py` — end-to-end pipeline entry point
- `config.yaml` — tickers, date range, optimization settings

## Usage
```
pip install -r requirements.txt
python main.py                 # uses config.yaml defaults
python main.py --no-push       # skip the post-run git commit
```

## Current status
Last run: (not yet run)
Tickers: (see config.yaml)
Covariance method: ledoit_wolf
Best Sharpe (tangency): N/A
Outputs: outputs/
