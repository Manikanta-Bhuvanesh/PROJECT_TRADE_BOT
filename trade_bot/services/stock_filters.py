"""Filter ``input/STOCKS.csv`` rows (sector / industry / cap) for engines; parse optional ``signal`` for the bot."""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Literal

import pandas as pd

from trade_bot.services.inputs import bot_stocks_path


def normalize_live_signal_side(raw: str | None) -> Literal["buy", "sell"] | None:
    """Accept ``buy`` / ``sell`` only (case-insensitive)."""
    if raw is None:
        return None
    s = str(raw).strip().lower()
    if s == "buy":
        return "buy"
    if s == "sell":
        return "sell"
    return None


def normalize_fib_flag(raw: str | None) -> bool | None:
    """Parse ``fib=true`` / ``false`` style values; unknown → ``None`` (ignore)."""
    if raw is None:
        return None
    s = str(raw).strip().lower()
    if s in ("1", "true", "yes", "on", "y"):
        return True
    if s in ("0", "false", "no", "off", "n"):
        return False
    return None


def filters_for_subprocess(filters: dict[str, str]) -> dict[str, str]:
    """Keys passed to engine subprocess (``signal`` / ``fib`` are bot-only for /live_signals)."""
    return {k: v for k, v in filters.items() if k not in ("signal", "fib")}


def parse_filters_from_command_args(args: list[str]) -> dict[str, str]:
    """
    Parse optional filters from Telegram command args.

    Supported keys: ``sector``, ``industry``, ``cap`` (aliases: ``marketcap``, ``marketcapname``),
    ``signal`` (``buy`` / ``sell``) for Telegram-side filtering of ``live_signals.csv`` only,
    and ``fib`` (``true`` / ``false``) for optional Fib columns on that CSV (bot-side only;
    Telegram bot: Fib off by default for everyone; pass ``fib=true`` to enable).

    Forms::

        /backtest_all sector=fmcg
        /live_signals cap=smallcap industry=chemicals
        /live_signals signal=buy
        /live_signals fib=true
        /live_signals signal=buy fib=true
        /backtest_all sector fmcg cap largecap

    ``cap`` values: ``largecap``, ``midcap``, ``smallcap`` (matched to ``marketcapname`` column).
    """
    out: dict[str, str] = {}
    i = 0
    a = [x.strip() for x in args if str(x).strip()]
    while i < len(a):
        raw = a[i]
        if "=" in raw:
            k, v = raw.split("=", 1)
            k = k.strip().lower()
            v = v.strip()
            if not v:
                i += 1
                continue
            if k in ("marketcap", "marketcapname"):
                k = "cap"
            if k == "signal":
                nv = normalize_live_signal_side(v)
                if nv:
                    out["signal"] = nv
                i += 1
                continue
            if k == "fib":
                nf = normalize_fib_flag(v)
                if nf is not None:
                    out["fib"] = "true" if nf else "false"
                i += 1
                continue
            if k in ("sector", "industry", "cap"):
                out[k] = v
            i += 1
            continue
        lk = raw.lower()
        if lk == "signal" and i + 1 < len(a):
            nv = normalize_live_signal_side(a[i + 1])
            if nv:
                out["signal"] = nv
            i += 2
            continue
        if lk == "fib" and i + 1 < len(a):
            nf = normalize_fib_flag(a[i + 1])
            if nf is not None:
                out["fib"] = "true" if nf else "false"
            i += 2
            continue
        if lk in ("sector", "industry", "cap", "marketcap", "marketcapname") and i + 1 < len(a):
            key = "cap" if lk in ("cap", "marketcap", "marketcapname") else lk
            out[key] = a[i + 1].strip()
            i += 2
            continue
        i += 1
    return out


def filters_to_subprocess_env(filters: dict[str, str]) -> dict[str, str]:
    """Env vars read by ``Algorithms.*.backtest_all`` / ``run_live_screen``."""
    ev: dict[str, str] = {}
    if v := (filters.get("sector") or "").strip():
        ev["STOCKBOT_FILTER_SECTOR"] = v
    if v := (filters.get("industry") or "").strip():
        ev["STOCKBOT_FILTER_INDUSTRY"] = v
    if v := (filters.get("cap") or "").strip():
        ev["STOCKBOT_FILTER_CAP"] = v.lower()
    return ev


def apply_stocks_csv_filters_from_env(df: pd.DataFrame) -> pd.DataFrame:
    """
    Subset ``df`` using ``STOCKBOT_FILTER_*`` env vars (set by the Telegram bot subprocess).

    - ``sector``: case-insensitive **exact** match on ``sector`` column (after strip).
    - ``industry``: case-insensitive **substring** match on ``industry`` (helps long Yahoo labels).
    - ``STOCKBOT_FILTER_CAP``: **exact** match on ``marketcapname`` (``largecap`` / ``midcap`` / ``smallcap``).
    """
    out = df.copy()
    sector = os.environ.get("STOCKBOT_FILTER_SECTOR", "").strip()
    industry = os.environ.get("STOCKBOT_FILTER_INDUSTRY", "").strip()
    cap = os.environ.get("STOCKBOT_FILTER_CAP", "").strip().lower()

    if sector:
        if "sector" not in out.columns:
            print("[filter] STOCKBOT_FILTER_SECTOR set but STOCKS.csv has no 'sector' column; ignoring.")
        else:
            s = out["sector"].astype(str).str.strip().str.lower()
            out = out.loc[s == sector.lower()].copy()
            print(f"[filter] sector={sector!r} -> {len(out)} rows")

    if industry:
        if "industry" not in out.columns:
            print("[filter] STOCKBOT_FILTER_INDUSTRY set but STOCKS.csv has no 'industry' column; ignoring.")
        else:
            s = out["industry"].astype(str).str.lower()
            needle = industry.lower()
            out = out.loc[s.str.contains(re.escape(needle), na=False)].copy()
            print(f"[filter] industry contains {industry!r} -> {len(out)} rows")

    if cap:
        col = "marketcapname"
        if col not in out.columns:
            print(f"[filter] STOCKBOT_FILTER_CAP set but STOCKS.csv has no {col!r} column; ignoring.")
        else:
            s = out[col].astype(str).str.strip().str.lower()
            out = out.loc[s == cap].copy()
            print(f"[filter] cap={cap!r} -> {len(out)} rows")

    return out


def distinct_column_sorted(project_root: Path, column: str) -> list[str]:
    p = bot_stocks_path(project_root)
    if not p.is_file():
        return []
    df = pd.read_csv(p)
    if column not in df.columns:
        return []
    ser = df[column].dropna().astype(str).str.strip()
    ser = ser[ser != ""]
    return sorted(ser.unique().tolist(), key=str.lower)
