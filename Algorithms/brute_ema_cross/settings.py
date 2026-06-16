"""
Configuration for ``brute_ema_cross`` (EMA crossover grid search on 15m bars).

Paths are resolved from the **project root** (parent of the ``Algorithms`` folder).

Data: Yahoo Finance 15-minute candles. Intraday history is capped by Yahoo; ``60d``
matches the typical maximum lookback for 15m data.
"""
from __future__ import annotations

import os
from pathlib import Path

# ── Paths (project root = parent of ``Algorithms``) ───────────────────────────
_ALGO_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _ALGO_DIR.parent.parent

INPUT_STOCKS_CSV: Path = _PROJECT_ROOT / "input" / "STOCKS.csv"
OUTPUT_DIR: Path = _PROJECT_ROOT / "output" / "brute_ema_cross"

# Legacy per-symbol JSON folder (optional); live loader prefers ``BEST_PARAMS_ALL_JSON``.
BEST_PARAMS_DIR: Path = _ALGO_DIR / "best_params"

# Single consolidated best-parameters file (used by ``signals.py``).
BEST_PARAMS_ALL_JSON: str = "best_params_all.json"

# Output filenames (under OUTPUT_DIR)
BACKTEST_ALL_CSV: str = "backtest_all_stocks.csv"
LIVE_SIGNALS_CSV: str = "live_signals.csv"

# ── Data fetch (yfinance via ``data_fetcher``) ─────────────────────────────────
# 15m intraday: use a bounded period (Yahoo caps intraday history; 60d is safe).
FETCH_PERIOD: str = "59d"
FETCH_INTERVAL: str = "15m"
# Smaller batches than daily — more rows per symbol per request.
FETCH_BATCH_SIZE: int = 12
FETCH_BATCH_PARALLEL_WORKERS: int = 8
FETCH_YF_THREADS: bool = True

# ── Parallel optimization ─────────────────────────────────────────────────────
PROCESS_POOL_MAX_WORKERS: int | None = None

# ── EMA span grid (15m; same structure as SMA). See ``brute_sma_cross`` settings
# for notes on tight pairs (e.g. 20 vs 25), step alignment, and MIN_SHORT_LONG_GAP.
SHORT_MA_MIN: int = 5
SHORT_MA_MAX: int = 50
SHORT_MA_STEP: int = 5

LONG_MA_MIN: int = 10
LONG_MA_MAX: int = 200
LONG_MA_STEP: int = 10

MIN_SHORT_LONG_GAP: int = 1

# Minimum 15m bars required (60 trading days × ~25 bars/day ≈ 1500 max; allow gaps)
MIN_PRICE_ROWS: int = 400

# ── Backtest / analyzer ───────────────────────────────────────────────────────
INITIAL_CAPITAL: float = 100_000.0
STOP_LOSS_PCT: float | None = None

SHARPE_WEIGHT: float = 15.0


def composite_score(total_return_pct: float, sharpe_ratio: float) -> float:
    return float(total_return_pct) + SHARPE_WEIGHT * float(sharpe_ratio)


def process_pool_workers() -> int:
    if PROCESS_POOL_MAX_WORKERS is not None and PROCESS_POOL_MAX_WORKERS > 0:
        return int(PROCESS_POOL_MAX_WORKERS)
    return min(32, max(1, (os.cpu_count() or 4)))


# ── Mass backtest guardrails ─────────────────────────────────────────────────
MAX_SYMBOLS: int | None = None

PROGRESS_EVERY: int = 50

# After ``run_live_screen()`` writes ``live_signals.csv``, send one email with the CSV
# attached when the file has at least one row. Requires env vars documented in
# ``notifications/live_signal_email.py`` (SMTP user/password/recipients).
EMAIL_ON_LIVE_SIGNALS: bool = False
