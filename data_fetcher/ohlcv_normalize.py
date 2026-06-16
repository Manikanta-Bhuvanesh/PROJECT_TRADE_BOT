"""
Normalize Yahoo Finance OHLCV frames (daily or intraday) for downstream use.

Algorithms and other callers use :func:`prepare_price_df` to get a sorted
``date`` + ``close`` series regardless of whether the index was a ``DatetimeIndex``
or the timestamp lived in a ``Datetime`` / ``Date`` column.
"""
from __future__ import annotations

import pandas as pd


def prepare_price_df(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize yfinance / OHLCV frames to ``date`` + ``close`` sorted ascending.

    Handles:

    - ``DatetimeIndex`` (daily or intraday): after ``reset_index()`` the first column may be
      named ``Datetime``, ``datetime``, ``Date``, ``Timestamp``, etc. — all mapped to ``date``.
    - Explicit ``Date`` / ``date`` columns.
    - Fallback: first column if it is datetime-like dtype.
    """
    if raw is None or raw.empty:
        return pd.DataFrame(columns=["date", "close"])

    d = raw.copy()

    if isinstance(d.index, pd.DatetimeIndex):
        d = d.reset_index()
        first = str(d.columns[0])
        if first.lower() != "date":
            d.rename(columns={first: "date"}, inplace=True)

    if "date" not in d.columns and "Date" in d.columns:
        d.rename(columns={"Date": "date"}, inplace=True)

    if "date" not in d.columns:
        lower = {str(c).lower(): c for c in d.columns}
        for alias in ("datetime", "timestamp", "timestamps"):
            if alias in lower:
                d.rename(columns={lower[alias]: "date"}, inplace=True)
                break

    if "date" not in d.columns and len(d.columns) > 0:
        cand = d.columns[0]
        try:
            if pd.api.types.is_datetime64_any_dtype(d[cand]):
                d.rename(columns={cand: "date"}, inplace=True)
        except (TypeError, ValueError):
            pass

    lower = {str(c).lower(): c for c in d.columns}
    close_col = lower.get("close")
    if close_col is None:
        raise ValueError(f"No Close column found. Columns: {list(d.columns)}")

    d = d.rename(columns={close_col: "close"})
    if "date" not in d.columns:
        raise ValueError(
            "Could not resolve a date column / DatetimeIndex. "
            f"Columns after normalization attempt: {list(d.columns)}"
        )

    out = d[["date", "close"]].copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    out.dropna(subset=["date", "close"], inplace=True)
    out.sort_values("date", kind="mergesort", inplace=True)
    out.reset_index(drop=True, inplace=True)
    return out


def prepare_ohlcv_df(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize yfinance OHLCV frames to ``date``, ``open``, ``high``, ``low``,
    ``close``, ``volume`` sorted ascending.

    Volume is coerced to numeric and NaNs filled with ``0.0`` so volume-based
    indicators behave like Pine ``nz(volume, 0)`` on missing points.
    """
    if raw is None or raw.empty:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])

    d = raw.copy()

    if isinstance(d.index, pd.DatetimeIndex):
        d = d.reset_index()
        first = str(d.columns[0])
        if first.lower() != "date":
            d.rename(columns={first: "date"}, inplace=True)

    if "date" not in d.columns and "Date" in d.columns:
        d.rename(columns={"Date": "date"}, inplace=True)

    if "date" not in d.columns:
        lower = {str(c).lower(): c for c in d.columns}
        for alias in ("datetime", "timestamp", "timestamps"):
            if alias in lower:
                d.rename(columns={lower[alias]: "date"}, inplace=True)
                break

    if "date" not in d.columns and len(d.columns) > 0:
        cand = d.columns[0]
        try:
            if pd.api.types.is_datetime64_any_dtype(d[cand]):
                d.rename(columns={cand: "date"}, inplace=True)
        except (TypeError, ValueError):
            pass

    lower = {str(c).lower(): c for c in d.columns}
    need = ("open", "high", "low", "close", "volume")
    missing = [x for x in need if x not in lower]
    if missing:
        raise ValueError(
            "Could not resolve OHLCV columns. "
            f"Missing: {missing}. Columns: {list(d.columns)}"
        )

    rename = {lower[x]: x for x in need}
    d = d.rename(columns=rename)

    out = d[["date", "open", "high", "low", "close", "volume"]].copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    for c in ("open", "high", "low", "close", "volume"):
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out["volume"] = out["volume"].fillna(0.0)
    out.dropna(subset=["date", "open", "high", "low", "close"], inplace=True)
    out.sort_values("date", kind="mergesort", inplace=True)
    out.reset_index(drop=True, inplace=True)
    return out
