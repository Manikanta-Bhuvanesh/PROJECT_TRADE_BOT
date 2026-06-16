"""Helpers for writing analyzer-related CSVs (display only; does not change backtest math)."""

from __future__ import annotations

import pandas as pd


def round_numeric_for_csv_export(df: pd.DataFrame, *, decimals: int = 2) -> pd.DataFrame:
    """
    Return a copy of ``df`` with floating-point columns rounded to ``decimals`` places.

    Intended **only** for CSV export: integers, booleans, datetimes, and non-numeric
    object columns are left unchanged.
    """
    if df is None or df.empty:
        return df
    out = df.copy()
    for c in out.columns:
        ser = out[c]
        if pd.api.types.is_bool_dtype(ser):
            continue
        if pd.api.types.is_integer_dtype(ser):
            continue
        if pd.api.types.is_datetime64_any_dtype(ser):
            continue
        if pd.api.types.is_timedelta64_dtype(ser):
            continue
        if pd.api.types.is_float_dtype(ser):
            out[c] = ser.round(decimals)
            continue
        kind = getattr(ser.dtype, "kind", None)
        if kind == "f":
            out[c] = ser.round(decimals)
            continue
        if ser.dtype == object:
            num = pd.to_numeric(ser, errors="coerce")
            non_null = ser.notna()
            if not bool(non_null.any()):
                continue
            if (num.notna() == non_null).all():
                out[c] = num.round(decimals)
    return out
