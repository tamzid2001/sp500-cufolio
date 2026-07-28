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

## Paper-only portfolio execution

For execution, use the persistent worker on an always-on host rather than
GitHub Actions' best-effort scheduler:

```bash
set -a; . ./.env; set +a
python -m cufolio_cpu.trading_worker
```

It persists completed actions in `var/paper_worker_state.json`, buys the
prepared dated target immediately after the 9:30 AM New York open, corrects
drift every 15 minutes through 3:15 PM, prepares tomorrow's target at 3:45 PM,
and flattens every position at 3:55 PM. It is paper-only.

`daily_cycle` is a current-S&P-500, one-trading-day long-only strategy. A
separate pre-close target job downloads 60 calendar days of completed 15-minute
Alpaca bars for all 503 listed S&P 500 share classes. It runs the existing
purged forward-return model: 1/4/13/26-bar return and 26-bar volatility
features, a requested 500-trading-minute forecast horizon (34 bars, or 510
minutes), and three purged walk-forward validation folds. A ridge forecast and
fully-invested, capped long-only forecast-minus-10-times-forward-variance solve
then creates a maximum-20-name target, with a 10% maximum position weight.
The newest incomplete 15-minute bar is excluded, and only a target with a
successful model run is written.

The paper workflow runs every 15 minutes on weekdays, at minutes 3, 18, 33,
and 48 to avoid the top-of-hour Actions queue. It uses fractional day market
orders, keeps purchases cash-only, and ignores movement below both $1 and a
25 bps absolute portfolio-weight drift band. A changed daily target sells all
old target holdings first and waits for fills before making the new buys.

At 3:45 PM America/New_York on weekdays, a
separate Action solves and commits the dated target for the next market session
using only completed 15-minute bars.
Beginning 30 minutes before the close, the 15-minute loop reads that already
solved target, retains positions that overlap it, and uses Alpaca's
position-close endpoint only for positions absent from it. New names are bought
after the following open. This minimizes needless sell/buy churn across daily
portfolios; overlapping positions can span the session boundary. Use a
dedicated strategy account because removed manual positions would be treated as
non-target strategy holdings. The GitHub scheduler is best-effort and can be
delayed, so it cannot guarantee a pre-close exit; use an always-on/self-hosted
runner or broker-hosted scheduling when a guaranteed exit deadline is required.

Create a local `.env` from `.env.example` and set the two keys there (the file
is ignored by Git). Export them before a plan-only local preview:

```bash
set -a
. ./.env
set +a
python -m cufolio_cpu.paper_rebalance \
  --targets assets/paper_target_weights.csv \
  --report artifacts/paper-rebalance/local-plan.json
```

Run a current-S&P-500 paper preview locally:

```bash
python -m cufolio_cpu.daily_cycle \
  --mode paper \
  --targets assets/active_daily_target.csv \
  --target-status assets/active_daily_target_status.json \
  --report artifacts/daily-cycle/local-plan.json
```

Add `--execute` to submit the paper orders. `.github/workflows/paper-rebalance.yml`
passes that flag only on its scheduled runs; a manually dispatched run remains
plan-only unless its `execute` input is checked. Set `ALPACA_API_KEY` and
`ALPACA_SECRET_KEY` as repository Actions secrets; do not put them in a tracked
file. Concurrency is locked so cycles cannot overlap.

`assets/paper_target_weights.csv` remains available for a deliberate, ad-hoc
rebalance through `cufolio_cpu.paper_rebalance`; it is no longer the scheduled
strategy input.

### Live mode is deliberately manual

The scheduler is paper-only. To enable a live preview or order submission, set
the separate `ALPACA_LIVE_API_KEY` and `ALPACA_LIVE_SECRET_KEY` variables and
use both explicit safeguards:

```bash
python -m cufolio_cpu.prepare_daily_target \
  --mode live \
  --output assets/live_daily_target.csv \
  --status assets/live_daily_target_status.json

python -m cufolio_cpu.daily_cycle \
  --mode live \
  --allow-live-trading \
  --targets assets/live_daily_target.csv \
  --target-status assets/live_daily_target_status.json \
  --report artifacts/daily-cycle/live-plan.json
```

Only add `--execute` after reviewing the plan. Live credentials never fall back
to the paper variables. The manual-only hourly live workflow also requires the
literal `SUBMIT_LIVE_ORDERS` confirmation plus GitHub's `live-trading`
environment, which should be configured with required reviewers and separate
`ALPACA_LIVE_API_KEY` / `ALPACA_LIVE_SECRET_KEY` secrets. It has no cron
schedule, and the hourly paper workflow remains paper-only.

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

### Forward 500-trading-minute research

The manual yfinance workflow additionally runs a separate, research-only
forward-horizon model. It uses trailing intraday returns and volatility to rank
stocks by predicted forward log return, validates later time blocks with a
purge that excludes labels overlapping the test window, then applies a
fully-invested, long-only mean–variance allocation to the highest-ranked
candidates using predicted returns and historical forward-return covariance.
It writes:

```text
forward_500m_research/forward_500m_validation.csv
forward_500m_research/forward_500m_candidate_portfolio.csv
forward_500m_research/forward_500m_status.json
```

The requested horizon is 500 **trading** minutes. One-minute bars represent it
exactly; fifteen-minute bars require 34 bars, so the output explicitly reports
an effective 510-minute horizon. The candidate portfolio is generated only
after 20 actual market sessions and at least two purged validation folds. It is
research output, not an optimal personal portfolio, a buy recommendation, or
an order instruction.

The resulting target weights are research output only—not investment advice,
account-specific share quantities, or order instructions.

### One-hour targets with 15-minute rebalancing

`cufolio_cpu.hourly_intraday_backtest` is a separate, research-only historical
backtest for one-minute bars. It constructs exact same-session one-hour target
returns at 09:30, 10:30, ..., 14:30 New York time. For example, it holds the
09:30 selection through 10:30 and restores its target weights at 09:30, 09:45,
10:00, and 10:15; at 10:30 it selects a fresh portfolio for 10:30--11:30.

At every hourly decision it selects the highest trailing expected one-hour
returns and solves a capped, long-only mean--variance allocation from only
earlier **completed** one-hour outcomes. It records transaction costs and
one-way turnover at every 15-minute rebalance. A target is rejected rather
than filled with stale prices when a required one-minute endpoint is missing;
overnight returns are never included. The optimizer is optimal only under its
trailing mean/covariance assumptions, not a guarantee of future results.

```bash
python -m cufolio_cpu.hourly_intraday_backtest \
  --input data/alpaca_one_minute_bars.csv \
  --output-dir outputs/hourly_one_hour_backtest \
  --top-n 20 --max-weight 0.10 --min-training-scenarios 20
```

The manual **Hourly one-hour portfolio research backtest** Action can use the
provided Alpaca credentials to download read-only historical one-minute bars,
or run the exact same engine against deterministic synthetic data. It uploads
the hourly selections, 15-minute rebalance ledger, and data-quality status as
an artifact; it never submits an order.

### Five- and fifteen-minute forecast audits

The manual **Five-minute intraday forecast audit** and **Fifteen-minute
intraday forecast audit** Actions both download read-only Alpaca IEX one-minute
history for a completed session. Each creates a fresh long-only, capped
portfolio at every non-overlapping interval and records the exact next-interval
return without forward-filling a missing endpoint. Because a one-minute bar is
left-labeled, decisions use the preceding completed close: a completed regular
session therefore has 77 causal five-minute decisions (09:35–15:55) or 25
causal fifteen-minute decisions (09:45–15:45), rather than inventing a 16:00
minute bar.

The reports include the complete decision ledger, per-holding prediction and
realization rows, all-forecast expected totals, and an expected-versus-actual
comparison restricted to the exact same realized windows. A compounded actual
return is called a session result only at 100% exact realization coverage;
otherwise it is explicitly a partial diagnostic. When IEX lacks an endpoint
for a selected holding, the Action fetches a one-minute yfinance fallback only
for those missing holdings and only replaces an outcome when both endpoints
are present. It never changes the original forecast, weights, or an existing
IEX realization. The forecasts use a pooled
non-overlapping intraday return estimate with a same-clock-time component
shrunk toward the pooled mean; they are not trading instructions or a claim of
future performance.

The hourly runner's 20-session history guard should not be read as a generic
one-minute-data rule. It is sized for that runner's 120 hourly labels. The
five-minute audit instead requires at least 10 prior sessions and 500 complete
non-overlapping interval scenarios; the 15-minute audit requires at least 10
prior sessions and 200 scenarios. This retains a meaningful covariance sample
while avoiding the error of treating overlapping one-minute-derived returns as
independent observations.

## GitHub Actions

`CPU notebooks` runs on ordinary Ubuntu GitHub-hosted runners. It runs tests,
then executes every CPU notebook with deterministic synthetic data. It neither
installs nor connects to NVIDIA software or infrastructure.

This is research software with opt-in execution tooling, not investment advice.

## Attribution

The methodology and upstream notebook organization are derived from NVIDIA's
cuFOLIO blueprint. See [`NOTICE.md`](NOTICE.md) and [`LICENSE`](LICENSE).
