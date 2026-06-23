# PROJECT_TRADE_BOT

Self-contained project: **Telegram bot** plus a **vendored copy** of the SMA/EMA engines, `data_fetcher`, `stock_analyzer`, and optional `notifications` (no separate `PROJECT_TRADE_SCREENER` install path).

## Setup (Windows, local)

```powershell
cd $env:USERPROFILE\Desktop\PROJECT_TRADE_BOT
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Copy `.env.example` → `.env` and set **`TELEGRAM_BOT_TOKEN`**. Optional **`ALLOWED_USER_IDS`** (comma-separated Telegram user IDs). **`ADMIN_USER_ID`** defaults to `945784800` (that user always has access). If both `ALLOWED_USER_IDS` and the optional merged allowlist are empty, any Telegram user may use the bot.

Put your universe in **`input/STOCKS.csv`** (at least a `symbol` column).

Run:

```powershell
python -m trade_bot
```

## Layout

| Path | Role |
|------|------|
| `trade_bot/` | Telegram app (`app.py`, handlers, services) |
| `Algorithms/brute_sma_cross/` | Daily SMA grid + live screen |
| `Algorithms/brute_ema_cross/` | 15m EMA grid + live screen |
| `data_fetcher/` | Yahoo NSE/BSE OHLCV helpers |
| `stock_analyzer/` | `analyze_trades`, equity curve |
| `input/STOCKS.csv` | Symbol universe + metadata |
| `output/` | Written by engines (`backtest_all_stocks.csv`, `live_signals.csv`, …) |

`EMAIL_ON_LIVE_SIGNALS` is set to **False** in the vendored strategy settings so the bot does not require SMTP unless you turn it back on.

## Telegram commands

After `/start`, choose **SMA** or **EMA**.

| Command | What it does |
|--------|----------------|
| `/backtest_all` | Bulk grid search → `backtest_all_stocks.csv` (**administrator only**; zip/split if large) |
| `/live_signals` | Latest-bar screen → `live_signals.csv` (only one run at a time across all users; needs prior `/backtest_all`) |
| `/get_stocks` | Sends `input/STOCKS.csv` as a document |
| `/get_sectors` | Lists distinct `sector` values from `STOCKS.csv` (document if long) |
| `/get_industries` | Sends distinct `industry` values as `industries.txt` |
| `/bt1 SYMBOL` | One-symbol optimize + metrics + equity PNG + trades CSV |
| `/deep SYMBOL` | Same outputs using saved `/backtest_all` params (no re-optimize; needs prior `/backtest_all`) |
| `/strategy` | Change SMA/EMA |

**Filters for `/backtest_all` and `/live_signals`** (optional; omit to run on the full universe):

- `sector=…` — exact match on `sector` (case-insensitive), e.g. `/backtest_all sector=fmcg`
- `industry=…` — substring on `industry`, e.g. `/live_signals industry=pharma`
- `cap=…` — `marketcapname` in `STOCKS.csv`: `largecap`, `midcap`, or `smallcap`, e.g. `/backtest_all cap=smallcap`

**`/live_signals` only** (after the screen runs, the bot filters the CSV it sends):

- `signal=buy` or `signal=sell` — attachment contains only rows with that `signal` value. Omit for both buy and sell in one file.

You can also use two-token form: `sector fmcg`, `industry pharma`, `cap midcap`, `signal buy`.

Note: `signal=…` is ignored by `/backtest_all` (engine-only filters are sector/industry/cap).

CLI: set `STOCKBOT_FILTER_SECTOR`, `STOCKBOT_FILTER_INDUSTRY`, `STOCKBOT_FILTER_CAP` before running `python -m Algorithms...backtest_all` or `signals` if you want the same filtering without the bot.

Optional: **`ENRICH_COMPANY_NAMES=1`** in the environment to add `company_name` via Yahoo (slow on large lists).

## Updating the engine later

If you improve the original screener, copy these folders from that repo over the copies here:

`Algorithms/` (at least `brute_sma_cross` and `brute_ema_cross`), `data_fetcher/`, `stock_analyzer/`, and `notifications/` if you use email.

## Security

Never commit `.env`. If a bot token was exposed, revoke it in BotFather and set a new one in `.env` only.
