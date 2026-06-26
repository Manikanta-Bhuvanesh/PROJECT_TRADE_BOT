"""
Moneycontrol price API provider for Indian equity OHLCV.

Fetches history from priceapi.moneycontrol.com and returns Yahoo-compatible
DataFrames (DatetimeIndex + Open/High/Low/Close/Volume) for use with
``prepare_price_df`` / ``prepare_ohlcv_df``.
"""

from __future__ import annotations

import json
import random
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any, Callable

import pandas as pd

BASE_URL = "https://priceapi.moneycontrol.com/techCharts/indianMarket/stock/history"
FROM_TS_DEFAULT = 662688000
COUNTBACK_DEFAULT = 100_000_000
SESSIONS_PER_WORKER = 2
REQUEST_TIMEOUT = 25
MAX_RETRIES = 3
RETRY_BASE_DELAY = 0.4

DEFAULT_MC_INTRADAY_PERIOD = "1y"

_MC_INTERVAL_MAP: dict[str, str] = {
    "1d": "1D",
    "1day": "1D",
    "d": "1D",
    "1m": "1",
    "5m": "5",
    "15m": "15",
    "30m": "30",
    "60m": "60",
    "1h": "60",
}


def mc_resolution_for_interval(interval: str) -> str | None:
    """Map a yfinance-style interval string to a Moneycontrol resolution, if supported."""
    return _MC_INTERVAL_MAP.get(interval.strip().lower())


def is_mc_intraday_interval(interval: str) -> bool:
    """True when the interval is intraday and Moneycontrol can serve it."""
    resolution = mc_resolution_for_interval(interval)
    if resolution is None:
        return False
    return resolution not in ("1D", "D", "1W", "W", "1M", "M")


class SessionPool:
    """Small rotating pool; replace session on transient HTTP failures."""

    def __init__(self, size: int) -> None:
        self._idx = 0
        self._sessions: list[dict[str, Any]] = [self._new_session() for _ in range(size)]

    @staticmethod
    def _new_session() -> dict[str, Any]:
        return {"ua_idx": random.randint(0, 2), "failures": 0}

    def _headers(self, sess: dict[str, Any]) -> dict[str, str]:
        uas = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 Safari/605.1.15",
        ]
        return {
            "User-Agent": uas[sess["ua_idx"] % len(uas)],
            "Accept": "application/json,text/plain,*/*",
            "Referer": "https://www.moneycontrol.com/",
            "Accept-Language": "en-US,en;q=0.9",
        }

    def get(self) -> tuple[int, dict[str, Any]]:
        idx = self._idx % len(self._sessions)
        self._idx += 1
        return idx, self._sessions[idx]

    def replace(self, idx: int) -> None:
        self._sessions[idx] = self._new_session()

    def request(self, url: str) -> bytes:
        last_exc: Exception | None = None
        for attempt in range(MAX_RETRIES):
            idx, sess = self.get()
            req = urllib.request.Request(url, headers=self._headers(sess))
            try:
                with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
                    return resp.read()
            except urllib.error.HTTPError as exc:
                last_exc = exc
                sess["failures"] += 1
                if exc.code in (403, 429, 502, 503, 504):
                    self.replace(idx)
                    time.sleep(RETRY_BASE_DELAY * (attempt + 1) + random.uniform(0, 0.3))
                    continue
                raise
            except Exception as exc:
                last_exc = exc
                sess["failures"] += 1
                self.replace(idx)
                time.sleep(RETRY_BASE_DELAY * (attempt + 1))
                continue
        if last_exc:
            raise last_exc
        raise RuntimeError("request failed without exception")


_POOL: SessionPool | None = None


def _init_worker() -> None:
    global _POOL
    _POOL = SessionPool(SESSIONS_PER_WORKER)


def _as_naive_ist(ts: str | pd.Timestamp) -> pd.Timestamp:
    """Normalize to tz-naive IST for comparisons against MC bar indices."""
    out = pd.Timestamp(ts)
    if out.tz is not None:
        return out.tz_convert("Asia/Kolkata").tz_localize(None)
    return out


def _period_cutoff(period: str | None, *, intraday: bool) -> pd.Timestamp | None:
    if not period:
        return None
    p = period.strip().lower()
    if p == "max":
        return None
    now = _as_naive_ist(pd.Timestamp.now("Asia/Kolkata"))
    if not intraday:
        now = now.normalize()
    if p == "ytd":
        return pd.Timestamp(now.year, 1, 1)
    m = re.match(r"^(\d+)(d|wk|mo|y)$", p)
    if not m:
        return None
    n, unit = int(m.group(1)), m.group(2)
    delta = {
        "d": pd.Timedelta(days=n),
        "wk": pd.Timedelta(weeks=n),
        "mo": pd.Timedelta(days=30 * n),
        "y": pd.Timedelta(days=365 * n),
    }[unit]
    return now - delta


def _trim_frame(
    df: pd.DataFrame,
    *,
    period: str | None,
    start: str | pd.Timestamp | None,
    end: str | pd.Timestamp | None,
    intraday: bool,
) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.sort_index()
    out.index = pd.to_datetime(out.index)
    if start is not None:
        out = out[out.index >= _as_naive_ist(start)]
    if end is not None:
        out = out[out.index <= _as_naive_ist(end)]
    if start is None and period:
        cutoff = _period_cutoff(period, intraday=intraday)
        if cutoff is not None:
            out = out[out.index >= cutoff]
    return out


def _mc_response_to_frame(payload: dict[str, Any]) -> pd.DataFrame | None:
    if payload.get("s") != "ok":
        return None
    timestamps = payload.get("t") or []
    if not timestamps:
        return None
    idx = (
        pd.to_datetime(timestamps, unit="s", utc=True)
        .tz_convert("Asia/Kolkata")
        .tz_localize(None)
    )
    frame = pd.DataFrame(
        {
            "Open": payload.get("o", []),
            "High": payload.get("h", []),
            "Low": payload.get("l", []),
            "Close": payload.get("c", []),
            "Volume": payload.get("v", []),
        },
        index=idx,
    )
    frame.index.name = "Datetime"
    return frame


def _build_url(symbol: str, resolution: str, to_ts: int) -> str:
    return (
        f"{BASE_URL}?symbol={symbol}&resolution={resolution}"
        f"&from={FROM_TS_DEFAULT}&to={to_ts}&countback={COUNTBACK_DEFAULT}"
        f"&currencyCode=INR"
    )


def _fetch_one_symbol_worker(
    packed: tuple[str, str, str | None, str | pd.Timestamp | None, str | pd.Timestamp | None, int],
) -> tuple[str, pd.DataFrame | None, str | None]:
    """
    Picklable worker: fetch one symbol from Moneycontrol.

    Returns ``(symbol, frame_or_none, error_or_none)``.
    """
    global _POOL
    assert _POOL is not None

    symbol, resolution, period, start, end, to_ts = packed
    url = _build_url(symbol, resolution, to_ts)
    intraday = resolution not in ("1D", "D", "1W", "W", "1M", "M")
    try:
        body = _POOL.request(url)
        payload = json.loads(body.decode("utf-8"))
        frame = _mc_response_to_frame(payload)
        if frame is None or frame.empty:
            status = payload.get("s", "unknown")
            return symbol, None, f"moneycontrol status={status}"
        frame = _trim_frame(frame, period=period, start=start, end=end, intraday=intraday)
        if frame.empty:
            return symbol, None, "moneycontrol returned no rows after period trim"
        return symbol, frame, None
    except Exception as exc:
        return symbol, None, str(exc)


def fetch_moneycontrol_equities(
    symbols: list[str],
    *,
    resolution: str,
    period: str | None = None,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    workers: int = 16,
    on_symbol_done: Callable[[], None] | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    """
    Fetch OHLCV for plain NSE symbols via Moneycontrol in parallel.

    Returns ``(frames_by_symbol, errors_by_symbol)``.
    """
    if not symbols:
        return {}, {}

    to_ts = int(datetime.now(timezone.utc).timestamp()) + 86400
    packs = [(sym, resolution, period, start, end, to_ts) for sym in symbols]
    frames: dict[str, pd.DataFrame] = {}
    errors: dict[str, str] = {}

    pool_workers = max(1, min(int(workers), len(packs)))
    with ProcessPoolExecutor(max_workers=pool_workers, initializer=_init_worker) as executor:
        futures = [executor.submit(_fetch_one_symbol_worker, pack) for pack in packs]
        for fut in as_completed(futures):
            symbol, frame, err = fut.result()
            if frame is not None and not frame.empty:
                frames[symbol] = frame
            else:
                errors[symbol] = err or "unknown moneycontrol error"
            if on_symbol_done is not None:
                on_symbol_done()

    return frames, errors
