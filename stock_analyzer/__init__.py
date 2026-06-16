"""Backtest-style performance metrics from signal + close history."""

from .analyzer import analyze_trades
from .csv_export import round_numeric_for_csv_export

__all__ = ["analyze_trades", "round_numeric_for_csv_export"]
