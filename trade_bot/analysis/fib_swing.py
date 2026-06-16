"""
Fibonacci retracement levels anchored on the latest qualifying price swing.

Swing pivots use a symmetric fractal window (same idea as Bill Williams fractals).
A leg must meet a minimum **relative** move:

- Low → high: ``(high - low) / low >= min_leg_pct``
- High → low: ``(high - low) / high >= min_leg_pct``

Among legs that qualify, the **latest** is the one whose **end** pivot has the
largest bar index (closest to the end of the series).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

Direction = Literal["up", "down"]

DEFAULT_FIB_RATIOS: tuple[float, ...] = (
    0.0,
    0.236,
    0.382,
    0.5,
    0.618,
    0.786,
    1.0,
)


@dataclass(frozen=True)
class FibSwingResult:
    """Retracement levels for one completed swing leg."""

    direction: Direction
    swing_low_price: float
    swing_high_price: float
    swing_low_index: int
    swing_high_index: int
    swing_low_date: pd.Timestamp | None
    swing_high_date: pd.Timestamp | None
    levels: dict[float, float]

    def levels_labeled(self) -> dict[str, float]:
        return {f"{k:g}": v for k, v in self.levels.items()}


def _pivot_high_mask(high: np.ndarray, k: int) -> np.ndarray:
    n = len(high)
    out = np.zeros(n, dtype=bool)
    for i in range(k, n - k):
        hi = high[i]
        left_max = np.max(high[i - k : i])
        right_max = np.max(high[i + 1 : i + k + 1])
        if hi > left_max and hi > right_max:
            out[i] = True
    return out


def _pivot_low_mask(low: np.ndarray, k: int) -> np.ndarray:
    n = len(low)
    out = np.zeros(n, dtype=bool)
    for i in range(k, n - k):
        lo = low[i]
        left_min = np.min(low[i - k : i])
        right_min = np.min(low[i + 1 : i + k + 1])
        if lo < left_min and lo < right_min:
            out[i] = True
    return out


def _collect_pivots(high: np.ndarray, low: np.ndarray, k: int) -> list[tuple[int, str, float]]:
    ph = _pivot_high_mask(high, k)
    pl = _pivot_low_mask(low, k)
    pivots: list[tuple[int, str, float]] = []
    for i in range(len(high)):
        if ph[i]:
            pivots.append((i, "H", float(high[i])))
        if pl[i]:
            pivots.append((i, "L", float(low[i])))
    pivots.sort(key=lambda x: x[0])
    return pivots


def _merge_same_kind_pivots(pivots: list[tuple[int, str, float]]) -> list[tuple[int, str, float]]:
    if not pivots:
        return []
    merged: list[tuple[int, str, float]] = [pivots[0]]
    for idx, kind, price in pivots[1:]:
        li, lk, lp = merged[-1]
        if kind == lk:
            if kind == "H":
                if price > lp or (price == lp and idx > li):
                    merged[-1] = (idx, kind, price)
            else:
                if price < lp or (price == lp and idx > li):
                    merged[-1] = (idx, kind, price)
        else:
            merged.append((idx, kind, price))
    return merged


def _datetime_column_name(df: pd.DataFrame) -> str | None:
    if "date" in df.columns:
        return "date"
    if "Date" in df.columns:
        return "Date"
    lower = {str(c).lower(): c for c in df.columns}
    for alias in ("datetime", "timestamp", "timestamps"):
        if alias in lower:
            return lower[alias]
    if len(df.columns) > 0:
        cand = df.columns[0]
        try:
            if pd.api.types.is_datetime64_any_dtype(df[cand]):
                return str(cand)
        except (TypeError, ValueError):
            pass
    return None


def _timestamp_at_iloc(df: pd.DataFrame, date_col: str, i: int) -> pd.Timestamp | None:
    raw = df.iloc[i][date_col]
    ts = pd.to_datetime(raw, errors="coerce")
    if pd.isna(ts):
        return None
    return ts


def _fib_levels_for_leg(
    direction: Direction,
    low_price: float,
    high_price: float,
    ratios: tuple[float, ...],
) -> dict[float, float]:
    rng = high_price - low_price
    if rng <= 0 or not np.isfinite(rng):
        return {r: float("nan") for r in ratios}
    levels: dict[float, float] = {}
    if direction == "up":
        for r in ratios:
            levels[r] = high_price - r * rng
    else:
        for r in ratios:
            levels[r] = low_price + r * rng
    return levels


def latest_swing_fibonacci(
    df: pd.DataFrame,
    *,
    min_leg_pct: float = 0.05,
    pivot_bars_each_side: int = 2,
    fib_ratios: tuple[float, ...] = DEFAULT_FIB_RATIOS,
) -> FibSwingResult | None:
    if df is None or df.empty:
        return None

    d = df.copy()
    lower = {str(c).lower(): c for c in d.columns}
    for col in ("high", "low"):
        if col not in d.columns and col in lower:
            d.rename(columns={lower[col]: col}, inplace=True)
    if "high" not in d.columns or "low" not in d.columns:
        raise ValueError("latest_swing_fibonacci requires 'high' and 'low' columns.")

    high = d["high"].to_numpy(dtype=float, copy=False)
    low = d["low"].to_numpy(dtype=float, copy=False)
    n = len(high)
    k = int(pivot_bars_each_side)
    if k < 1:
        raise ValueError("pivot_bars_each_side must be >= 1")
    if n < 2 * k + 3:
        return None

    pivots = _merge_same_kind_pivots(_collect_pivots(high, low, k))
    if len(pivots) < 2:
        return None

    best: tuple[int, Direction, int, int, float, float] | None = None

    for (i0, k0, p0), (i1, k1, p1) in zip(pivots, pivots[1:]):
        if k0 == "L" and k1 == "H":
            direction: Direction = "up"
            lo_i, hi_i = i0, i1
            lo_p, hi_p = p0, p1
            if lo_p <= 0 or not np.isfinite(lo_p):
                continue
            leg_pct = (hi_p - lo_p) / lo_p
        elif k0 == "H" and k1 == "L":
            direction = "down"
            hi_i, lo_i = i0, i1
            hi_p, lo_p = p0, p1
            if hi_p <= 0 or not np.isfinite(hi_p):
                continue
            leg_pct = (hi_p - lo_p) / hi_p
        else:
            continue

        if leg_pct < min_leg_pct or not np.isfinite(leg_pct):
            continue

        end_idx = i1
        if best is None or end_idx > best[0]:
            best = (end_idx, direction, lo_i, hi_i, lo_p, hi_p)

    if best is None:
        return None

    _, direction, lo_i, hi_i, lo_p, hi_p = best
    levels = _fib_levels_for_leg(direction, lo_p, hi_p, fib_ratios)

    date_col = _datetime_column_name(d)
    low_dt = _timestamp_at_iloc(d, date_col, lo_i) if date_col else None
    high_dt = _timestamp_at_iloc(d, date_col, hi_i) if date_col else None

    return FibSwingResult(
        direction=direction,
        swing_low_price=lo_p,
        swing_high_price=hi_p,
        swing_low_index=lo_i,
        swing_high_index=hi_i,
        swing_low_date=low_dt,
        swing_high_date=high_dt,
        levels=levels,
    )
