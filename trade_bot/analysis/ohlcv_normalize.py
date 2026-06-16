from __future__ import annotations

import pandas as pd


def yfinance_to_fib_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize a yfinance history frame (Datetime index, Open/High/Low/Close) for ``fib_swing``.

    Output includes ``date`` (timezone-aware or naive preserved), ``open``, ``high``, ``low``, ``close``.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    d = df.copy()
    idx_name = str(d.index.name or "").lower()
    if idx_name in ("datetime", "date"):
        d = d.reset_index()
        first = d.columns[0]
        d = d.rename(columns={first: "date"})
    elif not any(str(c).lower() == "date" for c in d.columns):
        d = d.reset_index()
        first = d.columns[0]
        d = d.rename(columns={first: "date"})

    ren: dict[str, str] = {}
    for c in d.columns:
        cl = str(c).lower()
        if cl == "open":
            ren[c] = "open"
        elif cl == "high":
            ren[c] = "high"
        elif cl == "low":
            ren[c] = "low"
        elif cl == "close":
            ren[c] = "close"
    d.rename(columns=ren, inplace=True)

    if "date" in d.columns:
        d["date"] = pd.to_datetime(d["date"], errors="coerce")

    return d
