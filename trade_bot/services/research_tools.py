from __future__ import annotations

import math
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from trade_bot.analysis.fib_swing import FibSwingResult, latest_swing_fibonacci
from trade_bot.analysis.ohlcv_normalize import yfinance_to_fib_ohlcv
from trade_bot.analysis.screener_company import fetch_screener_snapshot
from trade_bot.analysis.screener_comprehensive_summary import comprehensive_screener_summary
from data_fetcher.indian_equity import fetch_indian_equity

_IST = ZoneInfo("Asia/Kolkata")


def _format_fib_ts(x: Any) -> str | None:
    """Human-readable bar time for chat (IST), not raw ISO-8601."""
    if x is None:
        return None
    try:
        ts = pd.Timestamp(x)
    except Exception:
        return str(x)
    if pd.isna(ts):
        return None
    try:
        if ts.tzinfo is None:
            ts = ts.tz_localize(_IST)
        else:
            ts = ts.tz_convert(_IST)
    except Exception:
        return str(x)
    if ts.hour == 0 and ts.minute == 0 and ts.second == 0 and ts.microsecond == 0:
        return ts.strftime("%Y-%m-%d IST")
    return ts.strftime("%Y-%m-%d %H:%M IST")


def _fib_result_to_dict(res: FibSwingResult) -> dict[str, Any]:
    def _ts(x: Any) -> str | None:
        return _format_fib_ts(x)

    levels = {}
    for k, v in res.levels_labeled().items():
        if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
            levels[k] = None
        else:
            levels[k] = round(float(v), 6)
    return {
        "direction": res.direction,
        "swing_low_price": float(res.swing_low_price),
        "swing_high_price": float(res.swing_high_price),
        "swing_low_index": int(res.swing_low_index),
        "swing_high_index": int(res.swing_high_index),
        "swing_low_date": _ts(res.swing_low_date),
        "swing_high_date": _ts(res.swing_high_date),
        "levels": levels,
    }


def compute_fib_package(
    symbol: str,
    *,
    min_leg_pct_daily: float = 0.05,
    min_leg_pct_intraday: float = 0.03,
) -> dict[str, Any]:
    """
    Latest swing Fibonacci retracements on **1d**, **15m**, and **5m** Yahoo bars
    (NSE first, BSE fallback via ``fetch_indian_equity``).
    """
    sym = symbol.strip().upper()
    specs: list[tuple[str, str | None, str, float]] = [
        ("1d", "2y", "1d", min_leg_pct_daily),
        ("15m", "60d", "15m", min_leg_pct_intraday),
        ("5m", "60d", "5m", min_leg_pct_intraday),
    ]
    out: dict[str, Any] = {"symbol": sym, "timeframes": {}}
    for label, period, interval, min_pct in specs:
        df, st = fetch_indian_equity(sym, period=period, interval=interval, threads=True, timeout=45.0)
        block: dict[str, Any] = {
            "yahoo_ticker": st.yahoo_ticker,
            "rows": st.rows,
            "ok": st.ok,
            "error": st.error,
        }
        if not st.ok or df is None or df.empty:
            out["timeframes"][label] = block
            continue
        fib_df = yfinance_to_fib_ohlcv(df)
        block["bars_used"] = len(fib_df)
        res = latest_swing_fibonacci(fib_df, min_leg_pct=min_pct)
        if res is None:
            block["fib"] = None
            block["fib_note"] = "No qualifying swing for these bars and min_leg_pct."
        else:
            block["fib"] = _fib_result_to_dict(res)
            block["min_leg_pct"] = min_pct
        out["timeframes"][label] = block
    return out


def fib_package_to_text(pkg: dict[str, Any]) -> str:
    lines: list[str] = [f"Fib swing - {pkg.get('symbol')}", ""]
    for tf in ("1d", "15m", "5m"):
        blk = (pkg.get("timeframes") or {}).get(tf) or {}
        lines.append(f"=== {tf} ===")
        if not blk.get("ok"):
            lines.append(f"  (no data) {blk.get('error') or ''}".rstrip())
            lines.append("")
            continue
        lines.append(f"  bars: {blk.get('bars_used')}  ticker: {blk.get('yahoo_ticker')}")
        fib = blk.get("fib")
        if not fib:
            lines.append(f"  {blk.get('fib_note', 'No fib result.')}")
            lines.append("")
            continue
        lines.append(f"  direction: {fib.get('direction')}")
        lines.append(
            f"  swing low: {fib.get('swing_low_price')} @ {fib.get('swing_low_date')}"
        )
        lines.append(
            f"  swing high: {fib.get('swing_high_price')} @ {fib.get('swing_high_date')}"
        )
        lv = fib.get("levels") or {}

        def _rk(k: str) -> float:
            try:
                return float(k)
            except ValueError:
                return 0.0

        lines.append("  levels:")
        for k in sorted(lv.keys(), key=_rk):
            lines.append(f"    {k}: {lv.get(k)}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def fetch_screener_bundle(symbol: str) -> str:
    """
    Returns one long plain-text report (for Telegram ``<pre>`` chunks).

    Built from screener.in HTML + Price/EPS PE block internally; no file attachments.
    """
    sym = symbol.strip().upper()
    snap = fetch_screener_snapshot(
        sym,
        include_pe_chart=True,
        chart_days=365,
        request_delay_s=0.35,
        include_documents=False,
    )
    return comprehensive_screener_summary(snap, prefer_view="consolidated")
