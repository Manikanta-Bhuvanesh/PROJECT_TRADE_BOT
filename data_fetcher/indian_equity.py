from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import pandas as pd
import yfinance as yf
from tqdm.auto import tqdm

from .moneycontrol_provider import (
    DEFAULT_MC_INTRADAY_PERIOD,
    fetch_moneycontrol_equities,
    is_mc_intraday_interval,
    mc_resolution_for_interval,
)

NSE_SUFFIX = ".NS"
BSE_SUFFIX = ".BO"
DEFAULT_MONEYCONTROL_WORKERS = 16


def _silence_yfinance_logs() -> None:
    """
    Mute yfinance's noisy ``possibly delisted`` / ``Failed download`` messages.

    yfinance routes these through its own ``yfinance`` logger at ERROR level.
    Raising the threshold to CRITICAL keeps real exceptions intact while removing
    the per-symbol spam from bulk fetches. Runs at import so it also applies in
    the spawned child processes that perform the actual downloads.
    """
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)


_silence_yfinance_logs()


@dataclass
class SymbolFetchStatus:
    plain_symbol: str
    ok: bool
    yahoo_ticker: str | None = None
    rows: int = 0
    error: str | None = None


@dataclass
class BatchFetchResult:
    """Result of fetching one or more plain Indian equity symbols."""

    data: dict[str, pd.DataFrame] = field(default_factory=dict)
    status: list[SymbolFetchStatus] = field(default_factory=list)


def _base_symbol(sym: str) -> str:
    s = sym.strip().upper()
    if s.endswith(NSE_SUFFIX):
        return s[: -len(NSE_SUFFIX)]
    if s.endswith(BSE_SUFFIX):
        return s[: -len(BSE_SUFFIX)]
    return s


def _dedupe_preserve_bases(symbols: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for raw in symbols:
        b = _base_symbol(raw)
        if b not in seen:
            seen.add(b)
            out.append(b)
    return out


def _chunked(items: list[str], size: int) -> list[list[str]]:
    if size <= 0:
        raise ValueError("batch_size must be positive")
    return [items[i : i + size] for i in range(0, len(items), size)]


def _split_yahoo_download_df(df: pd.DataFrame | None) -> dict[str, pd.DataFrame]:
    if df is None or df.empty:
        return {}
    if not isinstance(df.columns, pd.MultiIndex):
        return {}
    out: dict[str, pd.DataFrame] = {}
    for yahoo_ticker in df.columns.get_level_values(0).unique():
        key = str(yahoo_ticker)
        part = df[key].copy()
        out[key] = part
    return out


def _history_usable(df: pd.DataFrame | None) -> bool:
    if df is None or df.empty:
        return False
    if "Close" in df.columns:
        return bool(df["Close"].notna().any())
    return bool(df.notna().to_numpy().any())


def _download_batch(
    yahoo_tickers: list[str],
    *,
    period: str | None,
    interval: str,
    start: str | pd.Timestamp | None,
    end: str | pd.Timestamp | None,
    threads: bool,
    timeout: float | None,
) -> dict[str, pd.DataFrame]:
    if not yahoo_tickers:
        return {}
    kwargs: dict = {
        "tickers": yahoo_tickers,
        "interval": interval,
        "group_by": "ticker",
        "threads": threads,
        "progress": False,
        "auto_adjust": True,
        "multi_level_index": True,
    }
    if timeout is not None:
        kwargs["timeout"] = timeout
    if start is not None or end is not None:
        kwargs["start"] = start
        kwargs["end"] = end
    else:
        if period is None:
            raise ValueError("Provide either period or both start/end for yfinance.")
        kwargs["period"] = period

    raw = yf.download(**kwargs)
    return _split_yahoo_download_df(raw)


def _fetch_one_symbol_batch(
    packed: tuple[
        tuple[str, ...],
        str | None,
        str,
        str | pd.Timestamp | None,
        str | pd.Timestamp | None,
        bool,
        float | None,
    ],
) -> tuple[dict[str, pd.DataFrame], dict[str, str], int]:
    """
    Picklable worker: one ``yf.download`` batch (NSE first, BSE fallback per symbol).

    Returns ``(frames_by_base, resolved_yahoo_ticker, n_symbols_in_batch)`` for tqdm.
    """
    _silence_yfinance_logs()
    batch_bases, period, interval, start, end, threads, timeout = packed
    frames: dict[str, pd.DataFrame] = {}
    resolved_yahoo: dict[str, str] = {}

    pending_ns: dict[str, str] = {b: f"{b}{NSE_SUFFIX}" for b in batch_bases}
    ns_list = [pending_ns[b] for b in batch_bases]
    ns_parts = _download_batch(
        ns_list,
        period=period,
        interval=interval,
        start=start,
        end=end,
        threads=threads,
        timeout=timeout,
    )

    need_bo: list[str] = []
    for base in batch_bases:
        y_ns = pending_ns[base]
        df_ns = ns_parts.get(y_ns)
        if _history_usable(df_ns):
            frames[base] = df_ns  # type: ignore[arg-type]
            resolved_yahoo[base] = y_ns
        else:
            need_bo.append(base)

    if need_bo:
        bo_list = [f"{b}{BSE_SUFFIX}" for b in need_bo]
        bo_parts = _download_batch(
            bo_list,
            period=period,
            interval=interval,
            start=start,
            end=end,
            threads=threads,
            timeout=timeout,
        )
        for base in need_bo:
            y_bo = f"{base}{BSE_SUFFIX}"
            df_bo = bo_parts.get(y_bo)
            if _history_usable(df_bo):
                frames[base] = df_bo  # type: ignore[arg-type]
                resolved_yahoo[base] = y_bo

    return frames, resolved_yahoo, len(batch_bases)


def _fetch_bases_via_yfinance(
    bases: list[str],
    *,
    period: str | None,
    interval: str,
    start: str | pd.Timestamp | None,
    end: str | pd.Timestamp | None,
    batch_size: int,
    threads: bool,
    timeout: float | None,
    batch_parallel_workers: int,
    show_progress: bool,
    bar: tqdm | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    """Download symbols via batched yfinance (existing behavior)."""
    if not bases:
        return {}, {}

    frames: dict[str, pd.DataFrame] = {}
    resolved_yahoo: dict[str, str] = {}

    batches = _chunked(bases, batch_size)
    packs = [
        (tuple(batch), period, interval, start, end, threads, timeout) for batch in batches
    ]

    workers = max(1, int(batch_parallel_workers))
    if workers <= 1 or len(packs) <= 1:
        for pack in packs:
            if bar is not None:
                bar.set_postfix_str("yfinance")
            f_part, r_part, n = _fetch_one_symbol_batch(pack)
            frames.update(f_part)
            resolved_yahoo.update(r_part)
            if bar is not None:
                bar.update(n)
    else:
        pool_workers = min(workers, len(packs))
        if bar is not None:
            bar.set_postfix_str(f"yfinance ({pool_workers} batches)")
        with ProcessPoolExecutor(max_workers=pool_workers) as ex:
            for f_part, r_part, n in ex.map(_fetch_one_symbol_batch, packs, chunksize=1):
                frames.update(f_part)
                resolved_yahoo.update(r_part)
                if bar is not None:
                    bar.update(n)

    return frames, resolved_yahoo


def _resolve_moneycontrol_period(
    interval: str,
    period: str | None,
    moneycontrol_period: str | None,
) -> str | None:
    """Moneycontrol intraday history is ~1y; yfinance fallback uses ``period``."""
    if moneycontrol_period is not None:
        return moneycontrol_period
    if is_mc_intraday_interval(interval):
        return DEFAULT_MC_INTRADAY_PERIOD
    return period


def fetch_indian_equity(
    symbol: str,
    *,
    period: str | None = "1mo",
    interval: str = "1d",
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    threads: bool = True,
    timeout: float | None = 30.0,
    use_moneycontrol: bool = True,
    moneycontrol_workers: int = DEFAULT_MONEYCONTROL_WORKERS,
    moneycontrol_period: str | None = None,
) -> tuple[pd.DataFrame, SymbolFetchStatus]:
    """Fetch one plain symbol: Moneycontrol first (when supported), then Yahoo NSE/BSE."""
    res = fetch_indian_equities(
        [symbol],
        period=period,
        interval=interval,
        start=start,
        end=end,
        batch_size=1,
        show_progress=False,
        threads=threads,
        timeout=timeout,
        use_moneycontrol=use_moneycontrol,
        moneycontrol_workers=moneycontrol_workers,
        moneycontrol_period=moneycontrol_period,
    )
    base = _base_symbol(symbol)
    df = res.data.get(base, pd.DataFrame())
    st = next((s for s in res.status if s.plain_symbol == base), SymbolFetchStatus(base, False, error="unknown"))
    return df, st


def fetch_indian_equities(
    symbols: Iterable[str],
    *,
    period: str | None = "1mo",
    interval: str = "1d",
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    batch_size: int = 10,
    show_progress: bool = True,
    threads: bool = True,
    timeout: float | None = 30.0,
    batch_parallel_workers: int = 1,
    use_moneycontrol: bool = True,
    moneycontrol_workers: int = DEFAULT_MONEYCONTROL_WORKERS,
    moneycontrol_period: str | None = None,
) -> BatchFetchResult:
    """
    Download OHLCV for plain Indian symbols.

    When ``use_moneycontrol`` is True and the interval is supported (e.g. ``1d``,
    ``15m``), symbols are fetched in parallel from Moneycontrol
    (``moneycontrol_workers``, default 16). Any misses are retried via batched
    yfinance (NSE then BSE), preserving the original batch download behavior.

    For intraday intervals (``1m``, ``5m``, ``15m``, …), Moneycontrol uses
    ``moneycontrol_period`` (default ``1y`` — MC's ~1-year intraday cap) while
    yfinance fallback always uses ``period`` (e.g. ``59d``).

    Parameters
    ----------
    symbols
        Plain symbols (e.g. ``RELIANCE``); optional ``.NS`` / ``.BO`` are stripped to base.
    period
        yfinance fallback window when ``start``/``end`` are not used (e.g. ``59d``).
    interval
        Bar size (e.g. ``1d``, ``15m``).
    start, end
        Optional bounds; if set, ``period`` is ignored for trimming.
    batch_size
        yfinance fallback: symbols per ``yf.download`` batch.
    show_progress
        Show a tqdm progress bar over symbol completion.
    threads
        Passed to yfinance fallback (intra-batch parallel fetches when True).
    timeout
        Per-request timeout for yfinance fallback (seconds).
    batch_parallel_workers
        yfinance fallback: concurrent batch downloads.
    use_moneycontrol
        Try Moneycontrol before yfinance when the interval is supported.
    moneycontrol_workers
        Parallel Moneycontrol fetch processes (default 16).
    moneycontrol_period
        Moneycontrol lookback (default ``1y`` for intraday intervals).
    """
    bases = _dedupe_preserve_bases(list(symbols))
    result = BatchFetchResult()
    if not bases:
        return result

    frames: dict[str, pd.DataFrame] = {}
    resolved_yahoo: dict[str, str] = {}
    fetch_errors: dict[str, str] = {}
    mc_resolution = mc_resolution_for_interval(interval) if use_moneycontrol else None
    mc_period = _resolve_moneycontrol_period(interval, period, moneycontrol_period)

    bar = tqdm(
        total=len(bases),
        desc="Symbols",
        unit="sym",
        disable=not show_progress,
        leave=True,
        ascii=True,
    )

    failed_bases = list(bases)
    if mc_resolution is not None:
        if show_progress:
            bar.set_postfix_str(f"moneycontrol ({moneycontrol_workers} workers)")
        mc_frames, mc_errors = fetch_moneycontrol_equities(
            bases,
            resolution=mc_resolution,
            period=mc_period,
            start=start,
            end=end,
            workers=moneycontrol_workers,
            on_symbol_done=bar.update if show_progress else None,
        )
        frames.update(mc_frames)
        fetch_errors.update(mc_errors)
        for base in mc_frames:
            resolved_yahoo[base] = f"{base}{NSE_SUFFIX}"
        failed_bases = [
            b for b in bases if b not in frames or not _history_usable(frames.get(b))
        ]

    if failed_bases:
        yf_frames, yf_resolved = _fetch_bases_via_yfinance(
            failed_bases,
            period=period,
            interval=interval,
            start=start,
            end=end,
            batch_size=batch_size,
            threads=threads,
            timeout=timeout,
            batch_parallel_workers=batch_parallel_workers,
            show_progress=show_progress,
            bar=bar if mc_resolution is None else None,
        )
        frames.update(yf_frames)
        resolved_yahoo.update(yf_resolved)

    for base in bases:
        if base in frames and _history_usable(frames[base]):
            df = frames[base]
            result.status.append(
                SymbolFetchStatus(
                    plain_symbol=base,
                    ok=True,
                    yahoo_ticker=resolved_yahoo.get(base),
                    rows=len(df),
                    error=None,
                )
            )
        else:
            mc_err = fetch_errors.get(base)
            yf_hint = "No data on NSE (.NS) or BSE (.BO) for this period/interval."
            if mc_err:
                error = f"{mc_err}; yfinance fallback: {yf_hint}"
            else:
                error = yf_hint
            result.status.append(
                SymbolFetchStatus(
                    plain_symbol=base,
                    ok=False,
                    yahoo_ticker=None,
                    rows=0,
                    error=error,
                )
            )

    result.data = frames
    if show_progress:
        bar.set_postfix_str("done")
        bar.close()

    return result
