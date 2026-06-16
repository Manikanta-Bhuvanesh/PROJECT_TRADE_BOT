from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path

from trade_bot.services.stock_filters import filters_to_subprocess_env
from trade_bot.services.subprocess_stream import run_python_module_streaming


def module_for_strategy(strategy: str) -> str:
    if strategy == "sma":
        return "Algorithms.brute_sma_cross.backtest_all"
    if strategy == "ema":
        return "Algorithms.brute_ema_cross.backtest_all"
    raise ValueError(f"Unknown strategy: {strategy!r}")


async def run_backtest_all_module_async(
    project_root: Path,
    strategy: str,
    *,
    on_progress: Callable[[str], Awaitable[None]] | None = None,
    mirror_terminal: bool = True,
    min_interval: float = 0.65,
    stock_filters: dict[str, str] | None = None,
) -> tuple[int, str]:
    """Async variant used by the Telegram bot (live log updates)."""
    extra = filters_to_subprocess_env(stock_filters or {})
    return await run_python_module_streaming(
        project_root,
        module_for_strategy(strategy),
        on_progress=on_progress,
        mirror_terminal=mirror_terminal,
        min_interval=min_interval,
        extra_env=extra if extra else None,
    )


def backtest_all_csv_path(project_root: Path, strategy: str) -> Path:
    if strategy == "sma":
        return project_root / "output" / "brute_sma_cross" / "backtest_all_stocks.csv"
    return project_root / "output" / "brute_ema_cross" / "backtest_all_stocks.csv"
