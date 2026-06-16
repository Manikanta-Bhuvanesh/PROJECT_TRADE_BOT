"""Indian equity OHLCV helpers built on yfinance (NSE/BSE suffix resolution).

Also exposes :func:`prepare_price_df` for normalizing downloaded frames (daily or intraday).
"""

from .indian_equity import (
    BatchFetchResult,
    SymbolFetchStatus,
    fetch_indian_equities,
    fetch_indian_equity,
)
from .ohlcv_normalize import prepare_ohlcv_df, prepare_price_df

__all__ = [
    "BatchFetchResult",
    "SymbolFetchStatus",
    "fetch_indian_equities",
    "fetch_indian_equity",
    "prepare_price_df",
    "prepare_ohlcv_df",
]
