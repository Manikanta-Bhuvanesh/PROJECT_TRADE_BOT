"""
Configuration for ``brute_sma_cross`` (SMA crossover grid search + screening).

Paths are resolved from the **project root** (parent of the ``Algorithms`` folder).
"""
from __future__ import annotations

import os
from pathlib import Path

# ── Paths (project root = parent of ``Algorithms``) ───────────────────────────
_ALGO_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _ALGO_DIR.parent.parent

INPUT_STOCKS_CSV: Path = _PROJECT_ROOT / "input" / "STOCKS.csv"
OUTPUT_DIR: Path = _PROJECT_ROOT / "output" / "brute_sma_cross"

# Legacy per-symbol JSON folder (optional); live loader prefers ``BEST_PARAMS_ALL_JSON``.
BEST_PARAMS_DIR: Path = _ALGO_DIR / "best_params"

# Single consolidated best-parameters file (used by ``signals.py``).
BEST_PARAMS_ALL_JSON: str = "best_params_all.json"

# Output filenames (under OUTPUT_DIR)
BACKTEST_ALL_CSV: str = "backtest_all_stocks.csv"
LIVE_SIGNALS_CSV: str = "live_signals.csv"

# ── Data fetch (yfinance via ``data_fetcher``) ─────────────────────────────────
FETCH_PERIOD: str = "max"
FETCH_INTERVAL: str = "1d"
FETCH_BATCH_SIZE: int = 100
# Concurrent yfinance *batches* (each batch = up to ``FETCH_BATCH_SIZE`` symbols).
# Example: 1000 symbols, batch 100 → 10 batches; workers 8 → up to 8 batches at once.
FETCH_BATCH_PARALLEL_WORKERS: int = 8
# yfinance ``threads`` for intra-batch parallel downloads (not ProcessPool).
FETCH_YF_THREADS: bool = True

# ── Parallel optimization ─────────────────────────────────────────────────────
# ``None`` → ``min(32, os.cpu_count() or 4)`` worker processes.
PROCESS_POOL_MAX_WORKERS: int | None = None

# ── SMA grid (daily crossover search) ─────────────────────────────────────────
# Only pairs with ``short < long`` are evaluated. Optional minimum gap below.
#
# Guidance (not enforced except MIN_SHORT_LONG_GAP):
# - Very close windows (e.g. 20 vs 25, gap 5) track almost the same curve → more
#   crosses and whipsaw on daily noise. Many practitioners use a larger spread
#   (e.g. long − short ≥ 10–20) or a clear ratio (long ≥ 1.5 × short).
# - With LONG_MA_STEP = 10, long candidates are 10, 20, 30, … so **(20, 25) never
#   appears** unless you set LONG_MA_STEP = 5 (or finer).
# - Finer steps → more combinations → slower backtests. Coarser steps → faster
#   but might skip a sharper optimum between grid points.
SHORT_MA_MIN: int = 5
SHORT_MA_MAX: int = 50
SHORT_MA_STEP: int = 5

LONG_MA_MIN: int = 10
LONG_MA_MAX: int = 200
LONG_MA_STEP: int = 10

# Require ``long - short >=`` this many bars (integer). ``1`` matches classic
# ``short < long`` only. Try ``10`` or ``15`` if you want to exclude tight pairs.
MIN_SHORT_LONG_GAP: int = 10

# Minimum rows of price data required to run the grid
MIN_PRICE_ROWS: int = 120

# ── Backtest / analyzer ───────────────────────────────────────────────────────
INITIAL_CAPITAL: float = 100_000.0
STOP_LOSS_PCT: float | None = None  # None = signal exits only

# Objective: maximize composite score (tune weights to taste)
SHARPE_WEIGHT: float = 15.0  # scales Sharpe roughly into %-like magnitude vs return


def composite_score(total_return_pct: float, sharpe_ratio: float) -> float:
    return float(total_return_pct) + SHARPE_WEIGHT * float(sharpe_ratio)


def process_pool_workers() -> int:
    if PROCESS_POOL_MAX_WORKERS is not None and PROCESS_POOL_MAX_WORKERS > 0:
        return int(PROCESS_POOL_MAX_WORKERS)
    return min(32, max(1, (os.cpu_count() or 4)))


# ── Mass backtest guardrails ─────────────────────────────────────────────────
# Set to a positive int to cap symbols (useful for smoke tests). None = all rows.
MAX_SYMBOLS: int | None = None

# Progress logging interval when processing many symbols
PROGRESS_EVERY: int = 50

# After ``run_live_screen()`` writes ``live_signals.csv``, send one email with the CSV
# attached when the file has at least one row. Requires env vars documented in
# ``notifications/live_signal_email.py`` (SMTP user/password/recipients).
EMAIL_ON_LIVE_SIGNALS: bool = False
