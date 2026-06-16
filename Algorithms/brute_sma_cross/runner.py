from __future__ import annotations

import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm.auto import tqdm

# Project root on sys.path for ``data_fetcher`` / ``stock_analyzer``
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data_fetcher import fetch_indian_equity, fetch_indian_equities, prepare_price_df  # noqa: E402
from stock_analyzer import analyze_trades  # noqa: E402

from . import settings  # noqa: E402
from .signals import sma_cross_signals  # noqa: E402


def _ensure_root_on_path() -> None:
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def _param_grid() -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    gap = max(1, int(getattr(settings, "MIN_SHORT_LONG_GAP", 1)))
    for s in range(settings.SHORT_MA_MIN, settings.SHORT_MA_MAX + 1, settings.SHORT_MA_STEP):
        for l in range(settings.LONG_MA_MIN, settings.LONG_MA_MAX + 1, settings.LONG_MA_STEP):
            if s < l and (l - s) >= gap:
                pairs.append((s, l))
    return pairs


def _scalar_backtest_metrics(result: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in result.items():
        if k in ("trades_detail", "equity_curve"):
            continue
        if v is None or isinstance(v, (str, int, float, bool)):
            out[k] = v
        elif isinstance(v, pd.DataFrame):
            continue
        else:
            try:
                out[k] = float(v)
            except (TypeError, ValueError):
                continue
    return out


def evaluate_sma_params(
    price: pd.DataFrame,
    short_w: int,
    long_w: int,
) -> dict[str, Any]:
    """Build signals and run ``analyze_trades``; returns flat metrics + windows."""
    sig_df = sma_cross_signals(price, short_w, long_w)
    bt = analyze_trades(
        sig_df,
        stop_loss_pct=settings.STOP_LOSS_PCT,
        initial_capital=settings.INITIAL_CAPITAL,
    )
    metrics = _scalar_backtest_metrics(bt)
    score = settings.composite_score(
        float(metrics.get("total_return_pct", 0.0)),
        float(metrics.get("sharpe_ratio", 0.0)),
    )
    return {
        "short_ma": short_w,
        "long_ma": long_w,
        "optimization_score": score,
        **metrics,
    }


def optimize_from_price(symbol: str, price: pd.DataFrame) -> dict[str, Any]:
    """
    Grid-search SMA lengths on an in-memory ``price`` frame (columns ``date``, ``close``).

    Returns a flat JSON-friendly dict including ``ok``, ``symbol``, best windows, and metrics.
    """
    sym = str(symbol).strip().upper()
    if price is None or price.empty or len(price) < settings.MIN_PRICE_ROWS:
        return {
            "symbol": sym,
            "ok": False,
            "error": f"insufficient_rows:{0 if price is None or price.empty else len(price)}",
            "best_short_ma": None,
            "best_long_ma": None,
            "optimization_score": None,
        }

    best: dict[str, Any] | None = None
    best_score = float("-inf")

    for sw, lw in _param_grid():
        row = evaluate_sma_params(price, sw, lw)
        sc = float(row.get("optimization_score", float("-inf")))
        if sc > best_score:
            best_score = sc
            best = row

    if best is None:
        return {
            "symbol": sym,
            "ok": False,
            "error": "no_grid",
            "best_short_ma": None,
            "best_long_ma": None,
            "optimization_score": None,
        }

    out: dict[str, Any] = {
        "symbol": sym,
        "ok": True,
        "error": None,
        "best_short_ma": int(best["short_ma"]),
        "best_long_ma": int(best["long_ma"]),
        "optimization_score": float(best_score),
    }
    for k, v in best.items():
        if k in ("short_ma", "long_ma", "optimization_score"):
            continue
        if v is None or isinstance(v, (str, int, float, bool)):
            out[k] = v
        else:
            try:
                fv = float(v)
                if math.isfinite(fv):
                    out[k] = fv
            except (TypeError, ValueError):
                pass
    return out


def _worker_optimize_one(packed: tuple[str, list[str], list[float]]) -> dict[str, Any]:
    """Picklable entry point for ``ProcessPoolExecutor`` (spawn-safe)."""
    _ensure_root_on_path()
    import pandas as pd

    sym, date_strs, closes = packed
    price = pd.DataFrame(
        {
            "date": pd.to_datetime(date_strs, errors="coerce"),
            "close": pd.to_numeric(closes, errors="coerce"),
        }
    )
    price.dropna(subset=["date", "close"], inplace=True)
    price.sort_values("date", kind="mergesort", inplace=True)
    price.reset_index(drop=True, inplace=True)

    return optimize_from_price(sym, price)


def bulk_fetch_prices(symbols: list[str]) -> dict[str, pd.DataFrame]:
    """
    Download **all** symbols in batched ``yf.download`` calls (threaded),
    returning **raw** OHLCV frames keyed by plain uppercase symbol.
    """
    res = fetch_indian_equities(
        symbols,
        period=settings.FETCH_PERIOD,
        interval=settings.FETCH_INTERVAL,
        batch_size=settings.FETCH_BATCH_SIZE,
        show_progress=True,
        threads=settings.FETCH_YF_THREADS,
        timeout=60.0,
        batch_parallel_workers=settings.FETCH_BATCH_PARALLEL_WORKERS,
    )
    return res.data


def pack_price_for_worker(raw: pd.DataFrame) -> tuple[list[str], list[float]] | None:
    """Serialize normalized price to plain lists for multiprocessing."""
    price = prepare_price_df(raw)
    if len(price) < settings.MIN_PRICE_ROWS:
        return None
    dates = [pd.Timestamp(x).isoformat() for x in price["date"].tolist()]
    closes = [float(x) for x in price["close"].tolist()]
    return dates, closes


def run_parallel_optimization(packed: list[tuple[str, list[str], list[float]]]) -> list[dict[str, Any]]:
    """Run ``optimize_from_price`` for each packed payload across a process pool."""
    if not packed:
        return []
    workers = settings.process_pool_workers()
    chunksize = max(1, len(packed) // max(1, workers * 8))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        it = ex.map(_worker_optimize_one, packed, chunksize=chunksize)
        return list(
            tqdm(
                it,
                total=len(packed),
                desc="SMA brute-force (process pool)",
                unit="sym",
                disable=False,
                ascii=True,
            )
        )


def build_best_params_document(
    *,
    stock_rows: pd.DataFrame,
    per_symbol_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Merge input universe metadata + optimization rows into one JSON document."""
    by_sym = {str(r["symbol"]).strip().upper(): r for r in per_symbol_results}
    results: dict[str, Any] = {}
    for sym, row in by_sym.items():
        base = dict(row)
        mrow = stock_rows[stock_rows["symbol"].astype(str).str.strip().str.upper() == sym]
        if not mrow.empty:
            mr = mrow.iloc[0]
            for col in ("marketcapname", "market_cap", "industry", "sector"):
                if col in mr.index:
                    base[col] = mr[col]
        results[sym] = base
    return {
        "meta": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "fetch_period": settings.FETCH_PERIOD,
            "fetch_interval": settings.FETCH_INTERVAL,
            "input_csv": str(settings.INPUT_STOCKS_CSV),
            "symbols_in_results": len(results),
        },
        "results": results,
    }


def save_best_params_all(doc: dict[str, Any], path: Path | None = None) -> Path:
    path = path or (settings.OUTPUT_DIR / settings.BEST_PARAMS_ALL_JSON)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.setdefault("meta", {})
    doc["meta"].setdefault("created_at", datetime.now(timezone.utc).isoformat())
    path.write_text(json.dumps(doc, indent=2, default=str), encoding="utf-8")
    return path


# ── Legacy / single-symbol helpers ────────────────────────────────────────────


def fetch_symbol_ohlcv(symbol: str) -> pd.DataFrame:
    """Download daily history for one plain symbol; returns raw OHLCV frame."""
    raw, st = fetch_indian_equity(
        symbol,
        period=settings.FETCH_PERIOD,
        interval=settings.FETCH_INTERVAL,
    )
    if not st.ok or raw is None or raw.empty:
        return pd.DataFrame()
    return raw


def optimize_symbol(symbol: str) -> dict[str, Any]:
    """
    One-off optimize (fetches data). Does **not** write per-symbol JSON anymore.
    """
    sym = str(symbol).strip().upper()
    raw = fetch_symbol_ohlcv(sym)
    price = prepare_price_df(raw)
    flat = optimize_from_price(sym, price)
    best = None
    if flat.get("ok"):
        best = {k: v for k, v in flat.items() if k not in ("symbol", "ok", "error")}
    return {
        "symbol": sym,
        "ok": bool(flat.get("ok")),
        "error": flat.get("error"),
        "best": best,
        "best_params_path": None,
    }


def best_row_for_csv(symbol: str, stock_row: pd.Series | None) -> dict[str, Any]:
    """Optimize one symbol (fetch) and flatten for CSV (legacy helper)."""
    res = optimize_symbol(symbol)
    base: dict[str, Any] = {"symbol": symbol}
    if stock_row is not None:
        for col in ("marketcapname", "market_cap", "industry", "sector"):
            if col in stock_row.index:
                base[col] = stock_row[col]

    if not res["ok"] or res["best"] is None:
        base["optimization_ok"] = False
        base["error"] = res.get("error")
        base["best_params_path"] = None
        return base

    b = res["best"] or {}
    base["optimization_ok"] = True
    base["error"] = None
    base["best_short_ma"] = b.get("best_short_ma")
    base["best_long_ma"] = b.get("best_long_ma")
    base["optimization_score"] = b.get("optimization_score")
    base["best_params_path"] = None
    for k, v in b.items():
        if k in ("best_short_ma", "best_long_ma", "optimization_score"):
            continue
        base[k] = v
    return base
