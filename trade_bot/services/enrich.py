from __future__ import annotations

from pathlib import Path

import pandas as pd

from stock_analyzer import round_numeric_for_csv_export


def merge_bot_stock_metadata(project_root: Path, csv_path: Path) -> None:
    """
    Re-attach columns from ``input/STOCKS.csv`` onto a results CSV
    (symbol, marketcapname, market_cap, industry, sector, …).
    """
    meta_src = project_root / "input" / "STOCKS.csv"
    stocks = pd.read_csv(meta_src)
    if "symbol" not in stocks.columns:
        raise ValueError("STOCKS.csv must include a symbol column.")
    out = pd.read_csv(csv_path)
    if "symbol" not in out.columns:
        return
    stocks = stocks.copy()
    stocks["symbol"] = stocks["symbol"].astype(str).str.strip().str.upper()
    out = out.copy()
    out["symbol"] = out["symbol"].astype(str).str.strip().str.upper()
    meta_cols = [c for c in stocks.columns if str(c).lower() != "symbol"]
    drop = [c for c in meta_cols if c in out.columns]
    if drop:
        out.drop(columns=drop, inplace=True)
    merged = out.merge(stocks, on="symbol", how="left")
    merged = round_numeric_for_csv_export(merged)
    merged.to_csv(csv_path, index=False)


def optional_company_names(csv_path: Path, *, max_symbols: int = 400) -> None:
    """
    Add ``company_name`` from Yahoo ``.NS`` when missing (capped for speed).
    """
    try:
        import yfinance as yf
    except ImportError:
        return

    df = pd.read_csv(csv_path)
    if "symbol" not in df.columns:
        return
    if "company_name" not in df.columns:
        df["company_name"] = pd.NA

    df["_symu"] = df["symbol"].astype(str).str.strip().str.upper()
    mask = df["company_name"].isna() | (df["company_name"].astype(str).str.strip() == "")
    candidates = df.loc[mask, "_symu"].tolist()[:max_symbols]
    for sym in candidates:
        try:
            info = yf.Ticker(f"{sym}.NS").info or {}
            name = info.get("longName") or info.get("shortName")
        except Exception:
            name = None
        if name:
            m = df["_symu"] == sym
            df.loc[m, "company_name"] = str(name)[:200]
    df.drop(columns=["_symu"], inplace=True, errors="ignore")
    df = round_numeric_for_csv_export(df)
    df.to_csv(csv_path, index=False)
