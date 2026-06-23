from __future__ import annotations

import io
import math
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def equity_curve_to_png(equity_df: pd.DataFrame, title: str) -> bytes:
    fig, ax = plt.subplots(figsize=(10, 4), dpi=120)
    d = pd.to_datetime(equity_df["date"], errors="coerce")
    ax.plot(d, equity_df["equity"].astype(float), color="#1f77b4", linewidth=1.2)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def _is_scalar_metric(v: Any) -> bool:
    if v is None or isinstance(v, bool):
        return True
    if isinstance(v, str):
        return True
    if isinstance(v, Real) and not isinstance(v, bool):
        try:
            x = float(v)
        except (TypeError, ValueError):
            return False
        return math.isfinite(x)
    return False


def format_all_scalar_metrics(merged: dict[str, Any]) -> str:
    """All finite scalar keys from optimization + backtest (sorted for stable output)."""
    lines: list[str] = []
    skip = frozenset({"ok", "error", "symbol"})
    for k in sorted(merged.keys(), key=str):
        if k in skip or k.startswith("_"):
            continue
        v = merged[k]
        if not _is_scalar_metric(v):
            continue
        lines.append(f"{k}: {v}")
    return "\n".join(lines) if lines else "(no scalar metrics)"


def format_stock_metadata_row(row: pd.Series | None) -> str:
    """Human-readable lines from ``input/STOCKS.csv`` (sector, industry, …)."""
    if row is None:
        return "(no row in input/STOCKS.csv for this symbol — add columns there for sector/industry/etc.)"
    skip = frozenset({"symbol", "symbol_key"})
    lines: list[str] = []
    for k in row.index:
        if str(k) in skip:
            continue
        v = row[k]
        try:
            if pd.isna(v):
                continue
        except TypeError:
            pass
        if v == "" or v is None:
            continue
        lines.append(f"{k}: {v}")
    return "\n".join(lines) if lines else "(no extra metadata columns in STOCKS.csv)"


def trades_to_vertical_preview(trades: pd.DataFrame, *, max_trades: int = 3) -> str:
    """
    One trade per block (readable on narrow Telegram screens).

    Shows the **most recent** ``max_trades`` rows; full history is sent as CSV.
    """
    if trades is None or trades.empty:
        return "No completed trades."
    n = int(len(trades))
    chunk = trades.tail(min(max_trades, n)).copy()
    cols = [c for c in chunk.columns if c not in ("entry_idx", "exit_idx")]
    start_num = n - len(chunk) + 1
    blocks: list[str] = []
    for j, (_, row) in enumerate(chunk.iterrows()):
        tnum = start_num + j
        lines = [f"--- trade {tnum}/{n} ---"]
        for c in cols:
            lines.append(f"  {c}: {row[c]}")
        blocks.append("\n".join(lines))
    intro = (
        f"(showing last {len(chunk)} of {n} completed trades — full table attached as CSV)\n\n"
        if n > len(chunk)
        else ""
    )
    return intro + "\n\n".join(blocks)


def trades_detail_to_csv_bytes(trades: pd.DataFrame) -> bytes | None:
    if trades is None or trades.empty:
        return None
    buf = io.StringIO()
    trades.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


@dataclass
class SingleStockAnalysis:
    symbol: str
    ok: bool
    error: str | None
    strategy_label: str
    best_short_ma: int | None
    best_long_ma: int | None
    optimization_score: float | None
    metrics_block: str
    equity_png: bytes | None
    last_trades_table: str
    latest_signal: str | None = None
    signal_bar_date: str | None = None
    signal_last_close: float | None = None
    stock_metadata_block: str = ""
    trades_csv: bytes | None = None
    completed_trade_count: int = 0


def _failed_analysis(
    *,
    sym: str,
    label: str,
    error: str,
    meta_txt: str,
) -> SingleStockAnalysis:
    return SingleStockAnalysis(
        symbol=sym,
        ok=False,
        error=error,
        strategy_label=label,
        best_short_ma=None,
        best_long_ma=None,
        optimization_score=None,
        metrics_block="",
        equity_png=None,
        last_trades_table="",
        latest_signal=None,
        signal_bar_date=None,
        signal_last_close=None,
        stock_metadata_block=meta_txt,
        trades_csv=None,
        completed_trade_count=0,
    )


def _strategy_bundle(strategy: str) -> tuple[Any, Callable, Callable, Callable, Callable, str]:
    if strategy == "sma":
        from Algorithms.brute_sma_cross import settings
        from Algorithms.brute_sma_cross.runner import fetch_symbol_ohlcv, optimize_from_price
        from Algorithms.brute_sma_cross.signals import load_best_parameters_table, sma_cross_signals

        return settings, fetch_symbol_ohlcv, optimize_from_price, load_best_parameters_table, sma_cross_signals, "SMA daily"
    if strategy == "ema":
        from Algorithms.brute_ema_cross import settings
        from Algorithms.brute_ema_cross.runner import fetch_symbol_ohlcv, optimize_from_price
        from Algorithms.brute_ema_cross.signals import ema_cross_signals, load_best_parameters_table

        return settings, fetch_symbol_ohlcv, optimize_from_price, load_best_parameters_table, ema_cross_signals, "EMA 15m"
    raise ValueError(strategy)


def _best_params_json_path(settings: Any) -> Path:
    return settings.OUTPUT_DIR / settings.BEST_PARAMS_ALL_JSON


def _load_saved_params_row(
    sym: str,
    load_best_parameters_table: Callable[[], pd.DataFrame],
    settings: Any,
) -> tuple[dict[str, Any] | None, str | None]:
    best = load_best_parameters_table()
    json_path = _best_params_json_path(settings)
    if best.empty:
        return None, (
            f"No saved parameters found (expected {json_path}). "
            "Run /backtest_all first."
        )

    if "symbol" not in best.columns:
        return None, f"Saved parameters file is missing a symbol column ({json_path})."

    best = best.copy()
    best["symbol"] = best["symbol"].astype(str).str.strip().str.upper()
    row_df = best[best["symbol"] == sym]
    if row_df.empty:
        return None, (
            f"`{sym}` is not in the saved backtest_all results ({json_path}). "
            "Run /backtest_all for this symbol (check sector/industry/cap filters)."
        )

    row = row_df.iloc[0]
    if row.get("ok") is False:
        err = row.get("error")
        return None, f"backtest_all failed for {sym}" + (f": {err}" if err else "")
    if "optimization_ok" in row.index and row["optimization_ok"] is False:
        return None, f"backtest_all could not optimize {sym} (insufficient data or grid)."

    try:
        sw = int(row["best_short_ma"])
        lw = int(row["best_long_ma"])
    except (TypeError, ValueError):
        return None, f"Saved results for {sym} are missing valid best_short_ma / best_long_ma."

    flat: dict[str, Any] = {
        "symbol": sym,
        "ok": True,
        "error": None,
        "best_short_ma": sw,
        "best_long_ma": lw,
        "optimization_score": float(row.get("optimization_score") or 0.0),
    }
    skip = frozenset(
        {"symbol", "ok", "error", "best_short_ma", "best_long_ma", "optimization_score", "optimization_ok"}
    )
    for k, v in row.items():
        if k in skip:
            continue
        try:
            if pd.isna(v):
                continue
        except TypeError:
            pass
        if _is_scalar_metric(v):
            flat[k] = v
    return flat, None


def _build_analysis_from_params(
    *,
    sym: str,
    price: pd.DataFrame,
    flat: dict[str, Any],
    settings: Any,
    sig_fn: Callable,
    meta_txt: str,
    label: str,
) -> SingleStockAnalysis:
    from stock_analyzer import analyze_trades

    if not flat.get("ok"):
        return _failed_analysis(sym=sym, label=label, error=str(flat.get("error")), meta_txt=meta_txt)

    sw = int(flat["best_short_ma"])
    lw = int(flat["best_long_ma"])
    sig_df = sig_fn(price, sw, lw)
    latest_signal = str(sig_df["signal"].iloc[-1]).lower()
    last_bar = sig_df.iloc[-1]
    bar_dt = pd.Timestamp(last_bar["date"])
    signal_bar_date = bar_dt.isoformat()
    signal_last_close = float(last_bar["close"])
    bt = analyze_trades(
        sig_df,
        stop_loss_pct=settings.STOP_LOSS_PCT,
        initial_capital=settings.INITIAL_CAPITAL,
    )
    eq = bt.get("equity_curve")
    png: bytes | None
    if isinstance(eq, pd.DataFrame) and not eq.empty:
        png = equity_curve_to_png(eq, f"{sym} — {label} (short={sw}, long={lw})")
    else:
        png = None

    td = bt.get("trades_detail")
    if not isinstance(td, pd.DataFrame):
        td = pd.DataFrame()

    merged_metrics = dict(flat)
    for k, v in bt.items():
        if k in ("trades_detail", "equity_curve"):
            continue
        if isinstance(v, (int, float, str, bool)) or v is None:
            merged_metrics[k] = v

    ret = float(merged_metrics.get("total_return_pct", 0.0) or 0.0)
    sharpe = float(merged_metrics.get("sharpe_ratio", 0.0) or 0.0)
    merged_metrics["optimization_score"] = settings.composite_score(ret, sharpe)

    csv_bytes = trades_detail_to_csv_bytes(td)
    n_done = int(len(td))

    return SingleStockAnalysis(
        symbol=sym,
        ok=True,
        error=None,
        strategy_label=label,
        best_short_ma=sw,
        best_long_ma=lw,
        optimization_score=float(merged_metrics.get("optimization_score") or 0.0),
        metrics_block=format_all_scalar_metrics(merged_metrics),
        equity_png=png,
        last_trades_table=trades_to_vertical_preview(td, max_trades=3),
        latest_signal=latest_signal,
        signal_bar_date=signal_bar_date,
        signal_last_close=signal_last_close,
        stock_metadata_block=meta_txt,
        trades_csv=csv_bytes,
        completed_trade_count=n_done,
    )


def analyze_single_stock(
    project_root: Path,
    strategy: str,
    symbol: str,
) -> SingleStockAnalysis:
    from data_fetcher import prepare_price_df
    from trade_bot.project_path import inject
    from trade_bot.services.inputs import stock_row_for_symbol

    inject(project_root)
    sym = str(symbol).strip().upper()
    settings, fetch_symbol_ohlcv, optimize_from_price, _, sig_fn, label = _strategy_bundle(strategy)

    stock_row = stock_row_for_symbol(project_root, sym)
    meta_txt = format_stock_metadata_row(stock_row)

    raw = fetch_symbol_ohlcv(sym)
    price = prepare_price_df(raw)
    flat = optimize_from_price(sym, price)

    if not flat.get("ok"):
        return _failed_analysis(sym=sym, label=label, error=str(flat.get("error")), meta_txt=meta_txt)

    return _build_analysis_from_params(
        sym=sym,
        price=price,
        flat=flat,
        settings=settings,
        sig_fn=sig_fn,
        meta_txt=meta_txt,
        label=label,
    )


def analyze_single_stock_from_saved_params(
    project_root: Path,
    strategy: str,
    symbol: str,
) -> SingleStockAnalysis:
    """Backtest one symbol using MA lengths from ``best_params_all.json`` (no grid search)."""
    from data_fetcher import prepare_price_df
    from trade_bot.project_path import inject
    from trade_bot.services.inputs import stock_row_for_symbol

    inject(project_root)
    sym = str(symbol).strip().upper()
    settings, fetch_symbol_ohlcv, _, load_best_parameters_table, sig_fn, label = _strategy_bundle(strategy)
    label = f"{label} (backtest_all params)"

    stock_row = stock_row_for_symbol(project_root, sym)
    meta_txt = format_stock_metadata_row(stock_row)

    flat, err = _load_saved_params_row(sym, load_best_parameters_table, settings)
    if flat is None:
        return _failed_analysis(sym=sym, label=label, error=err or "missing saved parameters", meta_txt=meta_txt)

    raw = fetch_symbol_ohlcv(sym)
    price = prepare_price_df(raw)
    if price is None or price.empty or len(price) < settings.MIN_PRICE_ROWS:
        n = 0 if price is None or price.empty else len(price)
        return _failed_analysis(
            sym=sym,
            label=label,
            error=f"insufficient price rows for backtest: {n} (need {settings.MIN_PRICE_ROWS})",
            meta_txt=meta_txt,
        )

    return _build_analysis_from_params(
        sym=sym,
        price=price,
        flat=flat,
        settings=settings,
        sig_fn=sig_fn,
        meta_txt=meta_txt,
        label=label,
    )
