# S&P 500 Portfolio Optimization — CPU Edition

This repository is a CPU-only, GitHub Actions runnable companion to NVIDIA's
[Quantitative Portfolio Optimization Blueprint](https://build.nvidia.com/nvidia/quantitative-portfolio-optimization).
It keeps the same practical workflow—prepare returns, construct Mean–CVaR or
mean–variance portfolios, inspect an efficient frontier, and backtest a
rebalancing rule—without requiring a GPU, NVIDIA cloud instance, CUDA, RAPIDS,
or cuOpt.

## What is copied and what runs

`upstream_notebooks/` contains unmodified copies of the NVIDIA cuFOLIO
notebooks at commit `dd7ca07db5e6c3624af80811f64562fc28480906`, retained under
Apache-2.0 with their original GPU requirements. `notebooks/` contains the
CPU-compatible counterparts generated from `tools/build_notebooks.py`; these
are what CI executes. The CPU versions use CVXPY with the open-source CLARABEL
solver and intentionally do not claim NVIDIA's GPU speed or scale.

## Input contract: one-minute bars to daily portfolio returns

The pipeline accepts a tidy minute-bar table:

```text
timestamp,symbol,close
2026-01-02T14:30:00Z,AAPL,250.10
2026-01-02T14:31:00Z,AAPL,250.25
```

It then:

1. converts timestamps to UTC and selects the US regular session
   (09:30–16:00 America/New_York);
2. computes intraday one-minute log returns by symbol **without crossing an
   overnight boundary**;
3. sums valid one-minute log returns into daily per-asset log returns;
4. converts these to daily simple returns for portfolio arithmetic; and
5. applies weights decided before a day to that day's simple returns.

Rows with fewer than the configurable `min_minutes_per_session` are marked
missing, not treated as zero returns. This avoids silently accepting incomplete
or stale intraday bars. `assets/universe.example.csv` is a small *format
example*, not a historical S&P 500 constituent file; replace it with a
point-in-time membership universe before a production backtest.

## Run locally

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
pytest
jupyter nbconvert --to notebook --execute notebooks/cvar_basic_cpu.ipynb \
  --output executed_cvar_basic.ipynb --ExecutePreprocessor.timeout=180
```

## Alpaca (optional; no credentials committed)

When you provide keys, place them in local environment variables, never in a
notebook or repository file:

```bash
export ALPACA_API_KEY='...'
export ALPACA_SECRET_KEY='...'
python -m cufolio_cpu.alpaca --symbols assets/universe.csv \
  --start 2026-01-02 --end 2026-01-31 --output data/minute_bars.csv
```

The GitHub Actions workflow always runs synthetic data in normal CI. Its manual
`alpaca` mode needs the two repository secrets, does not execute trades, and
only downloads minute bars then writes a build artifact. The data client follows
Alpaca's separate `StockHistoricalDataClient`/`StockBarsRequest` API surface.

## yfinance (credential-free research)

For a short recent research window, no Alpaca keys are required:

```bash
pip install -r requirements.txt
python -m cufolio_cpu.yfinance_data \
  --symbols assets/universe.example.csv \
  --start 2026-07-20 --end 2026-07-27 \
  --interval 1m \
  --output data/yfinance_intraday_bars.csv \
  --report data/yfinance_download_report.csv
python -m cufolio_cpu.research \
  --input data/yfinance_intraday_bars.csv --output-dir outputs/research
```

The manual `yfinance` workflow performs the same steps and uploads intraday
bars, daily log returns, daily simple returns, and `research_target_weights.csv`.
Use the `1m` interval for an eight-calendar-day maximum window, or `15m` for a
60-calendar-day maximum window. The workflow uses at least 300 one-minute
return observations or 20 fifteen-minute return observations per regular
session, respectively. In both modes it computes session returns from intraday
log returns and never crosses an overnight boundary.

Five market sessions validate the data pipeline but do not meet the 20-session
minimum for model weights; the workflow emits an explicit
`research_status.json` rather than a fabricated allocation. A 15-minute window
can provide enough daily observations for the *research* optimizer. It does not
by itself validate a prediction of the next 500 minutes or create a buy/order
instruction. The manual workflow can fetch a current 500-symbol convenience
universe from Wikipedia, but that universe is not point-in-time historical
membership and must not be used to claim a survivorship-bias-free backtest.
The resulting target weights are research output only—not investment advice,
account-specific share quantities, or order instructions.

## GitHub Actions

`CPU notebooks` runs on ordinary Ubuntu GitHub-hosted runners. It runs tests,
then executes every CPU notebook with deterministic synthetic data. It neither
installs nor connects to NVIDIA software or infrastructure.

This is research software, not investment advice or an execution system.

## Attribution

The methodology and upstream notebook organization are derived from NVIDIA's
cuFOLIO blueprint. See [`NOTICE.md`](NOTICE.md) and [`LICENSE`](LICENSE).
