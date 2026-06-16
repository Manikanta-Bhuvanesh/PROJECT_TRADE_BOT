from __future__ import annotations

from pathlib import Path

import pandas as pd


def bot_stocks_path(project_root: Path) -> Path:
    return project_root / "input" / "STOCKS.csv"


def ensure_input_stocks(project_root: Path) -> Path:
    """Ensure ``input/STOCKS.csv`` exists (vendored engine reads this path under project root)."""
    p = bot_stocks_path(project_root)
    if not p.is_file():
        raise FileNotFoundError(
            f"Missing {p}. Add input/STOCKS.csv (at least a `symbol` column; optional metadata columns)."
        )
    return p


def load_symbol_universe(project_root: Path) -> set[str]:
    p = ensure_input_stocks(project_root)
    df = pd.read_csv(p)
    if "symbol" not in df.columns:
        raise ValueError("STOCKS.csv must contain a 'symbol' column.")
    return set(df["symbol"].astype(str).str.strip().str.upper())


def stock_row_for_symbol(project_root: Path, symbol: str) -> pd.Series | None:
    sym = str(symbol).strip().upper()
    df = pd.read_csv(ensure_input_stocks(project_root))
    df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
    m = df[df["symbol"] == sym]
    if m.empty:
        return None
    return m.iloc[0]
