from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import pandas as pd
import yfinance as yf
from tqdm.auto import tqdm

NSE_SUFFIX = ".NS"
BSE_SUFFIX = ".BO"


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


def fetch_indian_equity(
    symbol: str,
    *,
    period: str | None = "1mo",
    interval: str = "1d",
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    threads: bool = True,
    timeout: float | None = 30.0,
) -> tuple[pd.DataFrame, SymbolFetchStatus]:
    """Fetch one plain symbol: try ``<BASE>.NS`` then ``<BASE>.BO``."""
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
) -> BatchFetchResult:
    """
    Download OHLCV for plain Indian symbols using yfinance.

    For each symbol, Yahoo tickers ``<BASE>.NS`` then ``<BASE>.BO`` are tried.
    Symbols are processed in batches of ``batch_size``; each batch uses one or two
    :func:`yfinance.download` calls (NSE batch, then BSE for misses).

    When ``batch_parallel_workers`` > 1, multiple batches run concurrently in
    child processes (``ProcessPoolExecutor``), up to
    ``min(batch_parallel_workers, number_of_batches)`` at a time — e.g. 1000 symbols,
    ``batch_size=100`` → 10 batches; with ``batch_parallel_workers=8``, up to eight
    batches download in parallel.

    Parameters
    ----------
    symbols
        Plain symbols (e.g. ``RELIANCE``); optional ``.NS`` / ``.BO`` are stripped to base.
    period
        yfinance period when ``start``/``end`` are not used (e.g. ``1mo``, ``1y``).
    interval
        Bar size (e.g. ``1d``, ``1h``).
    start, end
        Optional inclusive/exclusive bounds; if set, ``period`` is ignored.
    batch_size
        Number of distinct base symbols per ``yf.download`` batch.
    show_progress
        Show a tqdm progress bar over symbol completion.
    threads
        Passed to yfinance (intra-batch parallel fetches when True). With many
        ``batch_parallel_workers``, consider ``threads=False`` to reduce load.
    timeout
        Per-request timeout for yfinance (seconds); ``None`` for library default.
    batch_parallel_workers
        Maximum concurrent batch downloads. ``1`` keeps the original sequential
        behavior (lowest load on Yahoo).
    """
    bases = _dedupe_preserve_bases(list(symbols))
    result = BatchFetchResult()
    if not bases:
        return result

    resolved_yahoo: dict[str, str] = {}
    frames: dict[str, pd.DataFrame] = {}

    batches = _chunked(bases, batch_size)
    packs = [
        (tuple(batch), period, interval, start, end, threads, timeout) for batch in batches
    ]
    bar = tqdm(
        total=len(bases),
        desc="Symbols",
        unit="sym",
        disable=not show_progress,
        ascii=True,
        leave=True,
    )

    workers = max(1, int(batch_parallel_workers))
    if workers <= 1 or len(packs) <= 1:
        for pack in packs:
            if show_progress:
                bar.set_postfix_str("yf batch")
            f_part, r_part, n = _fetch_one_symbol_batch(pack)
            frames.update(f_part)
            resolved_yahoo.update(r_part)
            bar.update(n)
    else:
        pool_workers = min(workers, len(packs))
        if show_progress:
            bar.set_postfix_str(f"parallel batches ({pool_workers})")
        with ProcessPoolExecutor(max_workers=pool_workers) as ex:
            for f_part, r_part, n in ex.map(_fetch_one_symbol_batch, packs, chunksize=1):
                frames.update(f_part)
                resolved_yahoo.update(r_part)
                bar.update(n)

    # Status for every requested base (including failures)
    for base in bases:
        if base in frames:
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
            result.status.append(
                SymbolFetchStatus(
                    plain_symbol=base,
                    ok=False,
                    yahoo_ticker=None,
                    rows=0,
                    error="No data on NSE (.NS) or BSE (.BO) for this period/interval.",
                )
            )

    result.data = frames
    if show_progress:
        bar.set_postfix_str("done")
        bar.close()

    return result
