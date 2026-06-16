from __future__ import annotations

import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from data_fetcher import prepare_price_df

from . import settings


def _ensure_root_on_path() -> None:
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def _plain_scalar(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(v).isoformat()
    if isinstance(v, bool) or type(v) is np.bool_:
        return bool(v)
    if isinstance(v, (np.integer, int)) and not isinstance(v, bool):
        return int(v)
    if isinstance(v, (np.floating, float)):
        x = float(v)
        if math.isnan(x) or math.isinf(x):
            return None
        return x
    try:
        if pd.isna(v):
            return None
    except TypeError:
        pass
    return v


def _row_to_plain_dict(row: pd.Series) -> dict[str, Any]:
    """Pickle-friendly scalars for process-pool workers."""
    return {str(k): _plain_scalar(row[k]) for k in row.index}


def _worker_live_signal(
    packed: tuple[str, list[str], list[float], int, int, dict[str, Any], tuple[str, ...]],
) -> dict[str, Any] | None:
    """
    Picklable worker: rebuild OHLC from lists, apply best SMAs, return one CSV row
    if the latest bar is buy/sell; else ``None``.
    """
    _ensure_root_on_path()
    sym, date_strs, closes, sw, lw, merged_flat, col_order = packed

    price = pd.DataFrame(
        {
            "date": pd.to_datetime(date_strs, errors="coerce"),
            "close": pd.to_numeric(closes, errors="coerce"),
        }
    )
    price.dropna(subset=["date", "close"], inplace=True)
    price.sort_values("date", kind="mergesort", inplace=True)
    price.reset_index(drop=True, inplace=True)

    if len(price) < max(sw, lw) + 2:
        return None

    last_sig, sig_df = latest_signal(price, sw, lw)
    if last_sig not in ("buy", "sell"):
        return None

    last_dt = pd.Timestamp(sig_df["date"].iloc[-1])
    last_close = float(sig_df["close"].iloc[-1])

    rec: dict[str, Any] = {}
    for k in col_order:
        if k == "symbol_key":
            continue
        if k in merged_flat:
            rec[k] = merged_flat[k]
    rec["symbol"] = sym
    rec["best_short_ma"] = sw
    rec["best_long_ma"] = lw
    rec["signal"] = last_sig
    rec["signal_date"] = last_dt.isoformat()
    rec["last_close"] = last_close
    return rec


def _run_parallel_live_signals(
    jobs: list[tuple[str, list[str], list[float], int, int, dict[str, Any], tuple[str, ...]]],
) -> list[dict[str, Any] | None]:
    if not jobs:
        return []
    workers = settings.process_pool_workers()
    chunksize = max(1, len(jobs) // max(1, workers * 8))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        it = ex.map(_worker_live_signal, jobs, chunksize=chunksize)
        return list(
            tqdm(
                it,
                total=len(jobs),
                desc="Live signals (process pool)",
                unit="sym",
                disable=False,
                ascii=True,
            )
        )


def sma_cross_signals(price: pd.DataFrame, short_window: int, long_window: int) -> pd.DataFrame:
    """
    Classic SMA crossover signals on ``price`` (columns ``date``, ``close``).

    - **buy**  : prior bar ``SMA_short <= SMA_long`` and current ``SMA_short > SMA_long``
    - **sell** : prior bar ``SMA_short >= SMA_long`` and current ``SMA_short < SMA_long``
    - **hold** : otherwise

    Leading bars where SMAs are not yet valid are **hold**.
    """
    if short_window <= 0 or long_window <= 0:
        raise ValueError("SMA windows must be positive.")
    if short_window >= long_window:
        raise ValueError("SMA crossover expects short_window < long_window.")

    df = price.copy()
    c = df["close"].astype(float)
    sma_s = c.rolling(window=short_window, min_periods=short_window).mean()
    sma_l = c.rolling(window=long_window, min_periods=long_window).mean()

    sig: list[str] = []
    start = max(short_window, long_window)
    for i in range(len(df)):
        if i < start:
            sig.append("hold")
            continue
        ps, pl = float(sma_s.iloc[i - 1]), float(sma_l.iloc[i - 1])
        cs, cl = float(sma_s.iloc[i]), float(sma_l.iloc[i])
        if any(map(np.isnan, (ps, pl, cs, cl))):
            sig.append("hold")
            continue
        if ps <= pl and cs > cl:
            sig.append("buy")
        elif ps >= pl and cs < cl:
            sig.append("sell")
        else:
            sig.append("hold")

    out = pd.DataFrame({"date": df["date"], "close": c, "signal": sig})
    return out


def latest_signal(
    price: pd.DataFrame,
    short_window: int,
    long_window: int,
) -> tuple[Literal["buy", "sell", "hold"], pd.DataFrame]:
    """Return signal on the **last** row plus the full signal frame."""
    sig_df = sma_cross_signals(price, short_window, long_window)
    last = str(sig_df["signal"].iloc[-1]).lower()
    if last not in ("buy", "sell", "hold"):
        last = "hold"
    return last, sig_df  # type: ignore[return-value]


def load_best_parameters_table() -> pd.DataFrame:
    """
    Load best parameters as a flat table.

    Priority:
    1. ``output/brute_sma_cross/best_params_all.json`` (written by ``backtest_all``)
    2. ``output/brute_sma_cross/backtest_all_stocks.csv`` (legacy / spreadsheet)
    3. ``Algorithms/brute_sma_cross/best_params/*.json`` (legacy per-symbol files)
    """
    json_path = settings.OUTPUT_DIR / settings.BEST_PARAMS_ALL_JSON
    if json_path.is_file():
        doc = json.loads(json_path.read_text(encoding="utf-8"))
        rows: list[dict[str, Any]] = []
        for sym, row in (doc.get("results") or {}).items():
            if not isinstance(row, dict):
                continue
            r = dict(row)
            r.setdefault("symbol", sym)
            rows.append(r)
        return pd.DataFrame(rows)

    csv_path = settings.OUTPUT_DIR / settings.BACKTEST_ALL_CSV
    if csv_path.is_file():
        return pd.read_csv(csv_path)

    legacy_rows: list[dict[str, Any]] = []
    if settings.BEST_PARAMS_DIR.is_dir():
        for p in sorted(settings.BEST_PARAMS_DIR.glob("*.json")):
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            sym = str(d.get("symbol") or p.stem).strip()
            row: dict[str, Any] = {
                "symbol": sym,
                "best_short_ma": d.get("best_short_ma"),
                "best_long_ma": d.get("best_long_ma"),
                "optimization_score": d.get("optimization_score"),
                "optimization_ok": True,
            }
            for k, v in (d.get("metrics") or {}).items():
                row[k] = v
            legacy_rows.append(row)
    return pd.DataFrame(legacy_rows)


def run_live_screen() -> Path:
    """
    Screen symbols using saved best SMA lengths; emit rows where the **latest**
    bar is ``buy`` or ``sell``.

    1. Bulk-download all merged symbols once (``fetch_indian_equities``, threaded batches).
    2. ``ProcessPoolExecutor`` applies best SMAs per symbol and builds output rows in parallel.

    Output: ``output/brute_sma_cross/live_signals.csv`` (see ``settings``).
    """
    _ensure_root_on_path()
    from data_fetcher import fetch_indian_equities  # noqa: E402

    stocks = pd.read_csv(settings.INPUT_STOCKS_CSV)
    try:
        from trade_bot.services.stock_filters import apply_stocks_csv_filters_from_env

        stocks = apply_stocks_csv_filters_from_env(stocks)
    except Exception as exc:
        print(f"[filter] skipped: {exc}")
    if stocks.empty:
        out_path = settings.OUTPUT_DIR / settings.LIVE_SIGNALS_CSV
        settings.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame().to_csv(out_path, index=False)
        print("[warn] No stocks after STOCKBOT_FILTER_* — wrote empty live_signals.csv")
        return out_path
    stocks["symbol"] = stocks["symbol"].astype(str).str.strip().str.upper()
    best = load_best_parameters_table()
    if best.empty:
        raise RuntimeError(
            f"No best-parameters file found. Expected {settings.OUTPUT_DIR / settings.BEST_PARAMS_ALL_JSON}. "
            f"Run: python -m Algorithms.brute_sma_cross.backtest_all"
        )
    if "symbol" in best.columns:
        best["symbol"] = best["symbol"].astype(str).str.strip().str.upper()

    merged = stocks.merge(best, on="symbol", how="inner")

    candidates: list[tuple[pd.Series, int, int]] = []
    for _, row in merged.iterrows():
        if row.get("ok") is False:
            continue
        if "optimization_ok" in row.index and row["optimization_ok"] is False:
            continue
        try:
            sw = int(row["best_short_ma"])
            lw = int(row["best_long_ma"])
        except (TypeError, ValueError):
            continue
        candidates.append((row, sw, lw))

    symbols: list[str] = []
    seen: set[str] = set()
    for row, sw, lw in candidates:
        sym = str(row["symbol"]).strip().upper()
        if sym not in seen:
            seen.add(sym)
            symbols.append(sym)
    if not symbols:
        out_path = settings.OUTPUT_DIR / settings.LIVE_SIGNALS_CSV
        settings.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame().to_csv(out_path, index=False)
        return out_path

    print(
        f"[fetch] {len(symbols)} symbols, period={settings.FETCH_PERIOD!r}, "
        f"interval={settings.FETCH_INTERVAL!r} (batched download)"
    )
    batch = fetch_indian_equities(
        symbols,
        period=settings.FETCH_PERIOD,
        interval=settings.FETCH_INTERVAL,
        batch_size=settings.FETCH_BATCH_SIZE,
        show_progress=True,
        threads=settings.FETCH_YF_THREADS,
        timeout=60.0,
        batch_parallel_workers=settings.FETCH_BATCH_PARALLEL_WORKERS,
    )
    raw_map = batch.data

    jobs: list[tuple[str, list[str], list[float], int, int, dict[str, Any], tuple[str, ...]]] = []
    for row, sw, lw in tqdm(candidates, desc="Pack live-screen jobs", unit="sym", disable=False, ascii=True):
        sym = str(row["symbol"]).strip().upper()
        raw = raw_map.get(sym)
        if raw is None or getattr(raw, "empty", True):
            continue
        price = prepare_price_df(raw)
        if len(price) < max(sw, lw) + 2:
            continue
        dates = [pd.Timestamp(x).isoformat() for x in price["date"].tolist()]
        closes = [float(x) for x in price["close"].tolist()]
        merged_flat = _row_to_plain_dict(row)
        col_order = tuple(str(k) for k in row.index if str(k) != "symbol_key")
        jobs.append((sym, dates, closes, sw, lw, merged_flat, col_order))

    print(f"[optimize] workers={settings.process_pool_workers()} jobs={len(jobs)}")
    raw_results = _run_parallel_live_signals(jobs)
    out_rows = [r for r in raw_results if r is not None]

    settings.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = settings.OUTPUT_DIR / settings.LIVE_SIGNALS_CSV
    df_out = pd.DataFrame(out_rows)
    if not df_out.empty and out_rows:
        df_out = df_out[list(out_rows[0].keys())]
        from stock_analyzer import round_numeric_for_csv_export  # noqa: E402

        df_out = round_numeric_for_csv_export(df_out)
    df_out.to_csv(out_path, index=False)

    if getattr(settings, "EMAIL_ON_LIVE_SIGNALS", False):
        from notifications import mail_live_signals_csv_if_nonempty

        mail_live_signals_csv_if_nonempty(
            csv_path=out_path,
            strategy_label="SMA crossover (brute_sma_cross)",
        )

    return out_path


if __name__ == "__main__":
    out = run_live_screen()
    n = 0
    if out.exists() and out.stat().st_size > 0:
        n = len(pd.read_csv(out))
    print(f"Wrote {out} ({n} rows)")
