"""
Bulk-download all symbols once, brute-force best EMA spans per symbol in parallel
on **15m** bars (Moneycontrol **1y** lookback; yfinance fallback **59d**), and write
``best_params_all.json``.

Flow
----
1. Read ``input/STOCKS.csv`` symbol list.
2. ``fetch_indian_equities`` with ``moneycontrol_period=1y``, ``period=59d``,
   ``interval=15m`` (MC parallel fetch; yfinance batched fallback for misses).
3. ``ProcessPoolExecutor`` runs the EMA grid per symbol on in-memory prices.
4. Saves ``output/brute_ema_cross/best_params_all.json`` (used by ``signals.py``).
5. Saves ``output/brute_ema_cross/backtest_all_stocks.csv``.

Usage (from project root)::

    python -m Algorithms.brute_ema_cross.backtest_all
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
from tqdm.auto import tqdm

from Algorithms.brute_ema_cross import settings
from Algorithms.brute_ema_cross.runner import (
    build_best_params_document,
    bulk_fetch_prices,
    pack_price_for_worker,
    run_parallel_optimization,
    save_best_params_all,
)
from stock_analyzer import round_numeric_for_csv_export


def main() -> None:
    stocks = pd.read_csv(settings.INPUT_STOCKS_CSV)
    try:
        from trade_bot.services.stock_filters import apply_stocks_csv_filters_from_env

        stocks = apply_stocks_csv_filters_from_env(stocks)
    except Exception as exc:
        print(f"[filter] skipped: {exc}")
    if stocks.empty:
        print("[error] No rows left in STOCKS.csv after STOCKBOT_FILTER_* — abort.")
        sys.exit(2)
    if settings.MAX_SYMBOLS is not None:
        stocks = stocks.head(int(settings.MAX_SYMBOLS))

    symbols = stocks["symbol"].astype(str).str.strip().str.upper().tolist()
    print(
        f"[fetch] {len(symbols)} symbols, mc_period={settings.FETCH_MC_PERIOD!r}, "
        f"yf_period={settings.FETCH_PERIOD!r}, interval={settings.FETCH_INTERVAL!r} (tqdm in downloader)"
    )
    raw_map = bulk_fetch_prices(symbols)

    packed: list[tuple[str, list[str], list[float]]] = []
    failures: dict[str, dict] = {}

    for sym in tqdm(symbols, desc="Pack OHLCV for workers", unit="sym", disable=False, ascii=True):
        raw = raw_map.get(sym)
        if raw is None or getattr(raw, "empty", True):
            failures[sym] = {
                "symbol": sym,
                "ok": False,
                "error": "no_ohlcv",
                "best_short_ma": None,
                "best_long_ma": None,
                "optimization_score": None,
            }
            continue
        ser = pack_price_for_worker(raw)
        if ser is None:
            failures[sym] = {
                "symbol": sym,
                "ok": False,
                "error": "insufficient_rows",
                "best_short_ma": None,
                "best_long_ma": None,
                "optimization_score": None,
            }
            continue
        dates, closes = ser
        packed.append((sym, dates, closes))

    print(
        f"[optimize] workers={settings.process_pool_workers()} jobs={len(packed)} "
        "(progress bar below)"
    )
    parallel_rows = run_parallel_optimization(packed)
    by_sym = {str(r["symbol"]).strip().upper(): r for r in parallel_rows}

    ordered: list[dict] = []
    for sym in tqdm(symbols, desc="Order results", unit="sym", disable=False, ascii=True):
        if sym in by_sym:
            ordered.append(by_sym[sym])
        elif sym in failures:
            ordered.append(failures[sym])
        else:
            ordered.append(
                {
                    "symbol": sym,
                    "ok": False,
                    "error": "missing_result",
                    "best_short_ma": None,
                    "best_long_ma": None,
                    "optimization_score": None,
                }
            )

    doc = build_best_params_document(stock_rows=stocks, per_symbol_results=ordered)
    doc["meta"]["symbols_requested"] = len(symbols)
    doc["meta"]["symbols_fetched_ok"] = len(packed)
    doc["meta"]["symbols_ok_backtest"] = int(sum(1 for r in ordered if r.get("ok")))
    doc["meta"]["pipeline_finished_at"] = datetime.now(timezone.utc).isoformat()

    settings.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = save_best_params_all(doc)
    print(f"[write] {json_path}")

    sk = stocks.assign(symbol_key=stocks["symbol"].astype(str).str.strip().str.upper())
    dfk = pd.DataFrame(ordered).assign(
        symbol_key=lambda d: d["symbol"].astype(str).str.strip().str.upper()
    )
    drop_cols = [c for c in ("symbol",) if c in dfk.columns]
    merged = sk.merge(dfk.drop(columns=drop_cols, errors="ignore"), on="symbol_key", how="right")
    merged.drop(columns=["symbol_key"], inplace=True, errors="ignore")
    csv_path = settings.OUTPUT_DIR / settings.BACKTEST_ALL_CSV
    round_numeric_for_csv_export(merged).to_csv(csv_path, index=False)
    print(f"[write] {csv_path} ({len(merged)} rows)")


if __name__ == "__main__":
    main()
