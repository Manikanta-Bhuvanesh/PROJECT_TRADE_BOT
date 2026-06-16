from __future__ import annotations

import io
import math
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any

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


def trades_to_vertical_preview(trades: pd.DataFrame, *, max_trades: int = 15) -> str:
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


def analyze_single_stock(
    project_root: Path,
    strategy: str,
    symbol: str,
) -> SingleStockAnalysis:
    from trade_bot.project_path import inject

    inject(project_root)
    sym = str(symbol).strip().upper()

    if strategy == "sma":
        from Algorithms.brute_sma_cross import settings
        from Algorithms.brute_sma_cross.runner import fetch_symbol_ohlcv, optimize_from_price
        from Algorithms.brute_sma_cross.signals import sma_cross_signals

        sig_fn = sma_cross_signals
    elif strategy == "ema":
        from Algorithms.brute_ema_cross import settings
        from Algorithms.brute_ema_cross.runner import fetch_symbol_ohlcv, optimize_from_price
        from Algorithms.brute_ema_cross.signals import ema_cross_signals

        sig_fn = ema_cross_signals
    else:
        raise ValueError(strategy)

    from data_fetcher import prepare_price_df
    from stock_analyzer import analyze_trades
    from trade_bot.services.inputs import stock_row_for_symbol

    stock_row = stock_row_for_symbol(project_root, sym)
    meta_txt = format_stock_metadata_row(stock_row)

    raw = fetch_symbol_ohlcv(sym)
    price = prepare_price_df(raw)
    flat = optimize_from_price(sym, price)
    label = "SMA daily" if strategy == "sma" else "EMA 15m"

    if not flat.get("ok"):
        return SingleStockAnalysis(
            symbol=sym,
            ok=False,
            error=str(flat.get("error")),
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

    csv_bytes = trades_detail_to_csv_bytes(td)
    n_done = int(len(td))

    return SingleStockAnalysis(
        symbol=sym,
        ok=True,
        error=None,
        strategy_label=label,
        best_short_ma=sw,
        best_long_ma=lw,
        optimization_score=float(flat.get("optimization_score") or 0.0),
        metrics_block=format_all_scalar_metrics(merged_metrics),
        equity_png=png,
        last_trades_table=trades_to_vertical_preview(td, max_trades=15),
        latest_signal=latest_signal,
        signal_bar_date=signal_bar_date,
        signal_last_close=signal_last_close,
        stock_metadata_block=meta_txt,
        trades_csv=csv_bytes,
        completed_trade_count=n_done,
    )
