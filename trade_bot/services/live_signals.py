from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from stock_analyzer import round_numeric_for_csv_export
from trade_bot.services.stock_filters import filters_to_subprocess_env
from trade_bot.services.subprocess_stream import run_python_module_streaming

log = logging.getLogger(__name__)

_FIB_TFS = ("1d", "15m", "5m")


def fib_row_from_package(pkg: dict[str, Any]) -> dict[str, Any]:
    """Flat columns for CSV: per timeframe direction, swing high/low price + bar time (IST, no levels)."""
    out: dict[str, Any] = {}
    tfs = pkg.get("timeframes") or {}
    for tf in _FIB_TFS:
        prefix = f"fib_{tf}_"
        blk = tfs.get(tf) or {}
        fib = blk.get("fib")
        if not fib:
            out[f"{prefix}direction"] = ""
            out[f"{prefix}swing_high"] = ""
            out[f"{prefix}swing_high_at"] = ""
            out[f"{prefix}swing_low"] = ""
            out[f"{prefix}swing_low_at"] = ""
            continue
        out[f"{prefix}direction"] = str(fib.get("direction") or "")
        try:
            out[f"{prefix}swing_high"] = round(float(fib.get("swing_high_price")), 2)
        except (TypeError, ValueError):
            out[f"{prefix}swing_high"] = ""
        out[f"{prefix}swing_high_at"] = str(fib.get("swing_high_date") or "")
        try:
            out[f"{prefix}swing_low"] = round(float(fib.get("swing_low_price")), 2)
        except (TypeError, ValueError):
            out[f"{prefix}swing_low"] = ""
        out[f"{prefix}swing_low_at"] = str(fib.get("swing_low_date") or "")
    return out


def enrich_live_signals_csv_with_fib(csv_path: Path) -> int:
    """
    Append Fib swing summary columns (1d / 15m / 5m: direction, swing high/low price + time)
    and rewrite ``live_signals.csv`` in place.
    """
    from trade_bot.services.research_tools import compute_fib_package

    if not csv_path.is_file():
        return 0
    df = pd.read_csv(csv_path)
    if df.empty or "symbol" not in df.columns:
        return 0
    drop = [c for c in df.columns if str(c).startswith("fib_")]
    if drop:
        df = df.drop(columns=drop, errors="ignore")
    rows: list[dict[str, Any]] = []
    for sym in df["symbol"].astype(str).str.strip().str.upper():
        try:
            pkg = compute_fib_package(sym)
        except Exception as exc:
            log.warning("fib enrich failed for %s: %s", sym, exc)
            pkg = {"symbol": sym, "timeframes": {}}
        rows.append(fib_row_from_package(pkg))
    fib_part = pd.DataFrame(rows)
    merged = pd.concat([df.reset_index(drop=True), fib_part], axis=1)
    merged = round_numeric_for_csv_export(merged)
    merged.to_csv(csv_path, index=False)
    return len(merged)


def module_for_strategy(strategy: str) -> str:
    if strategy == "sma":
        return "Algorithms.brute_sma_cross.signals"
    if strategy == "ema":
        return "Algorithms.brute_ema_cross.signals"
    raise ValueError(strategy)


async def run_live_signals_module_async(
    project_root: Path,
    strategy: str,
    *,
    on_progress: Callable[[str], Awaitable[None]] | None = None,
    mirror_terminal: bool = True,
    min_interval: float = 0.65,
    stock_filters: dict[str, str] | None = None,
) -> tuple[int, str]:
    """
    Run ``python -m Algorithms.brute_*_cross.signals`` (same entrypoint as CLI).

    Returns ``(exit_code, combined_stdout_stderr)``. Output CSV path is implied by
    :func:`live_signals_csv_path` on success.
    """
    extra = filters_to_subprocess_env(stock_filters or {})
    return await run_python_module_streaming(
        project_root,
        module_for_strategy(strategy),
        on_progress=on_progress,
        mirror_terminal=mirror_terminal,
        min_interval=min_interval,
        extra_env=extra if extra else None,
    )


def live_signals_csv_path(project_root: Path, strategy: str) -> Path:
    if strategy == "sma":
        return project_root / "output" / "brute_sma_cross" / "live_signals.csv"
    return project_root / "output" / "brute_ema_cross" / "live_signals.csv"


def materialize_live_signals_upload_path(
    source_csv: Path,
    work_dir: Path,
    signal_side: Literal["buy", "sell"] | None,
) -> tuple[Path, bool]:
    """
    Return ``(path_to_upload, applied_side_filter)``.

    When ``signal_side`` is set and the CSV has a ``signal`` column, writes
    ``live_signals_<side>.csv`` under ``work_dir`` and returns that path with
    ``applied_side_filter=True``. Otherwise returns ``source_csv`` and ``False``.
    """
    if signal_side is None or not source_csv.is_file():
        return source_csv, False
    df = pd.read_csv(source_csv)
    if "signal" not in df.columns:
        return source_csv, False
    s = df["signal"].astype(str).str.strip().str.lower()
    df = df.loc[s == signal_side].copy()
    df = round_numeric_for_csv_export(df)
    work_dir.mkdir(parents=True, exist_ok=True)
    dest = work_dir / f"{source_csv.stem}_{signal_side}{source_csv.suffix}"
    df.to_csv(dest, index=False)
    return dest, True
