"""
Sample run: read symbols from ``input/STOCKS.csv`` and fetch the first 10
in a single batch (``batch_size=10``).

From project root::

    python data_fetcher/run_sample.py

Or::

    python -m data_fetcher.run_sample
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from data_fetcher.indian_equity import fetch_indian_equities  # noqa: E402


def main() -> None:
    csv_path = ROOT / "input" / "STOCKS.csv"
    symbols = pd.read_csv(csv_path)["symbol"].astype(str).head(10).tolist()
    print("First 10 symbols from input/STOCKS.csv:", symbols)
    res = fetch_indian_equities(
        symbols,
        period="1mo",
        interval="1d",
        batch_size=10,
        show_progress=True,
        threads=True,
        timeout=45.0,
    )
    print("\n--- status ---")
    for st in res.status:
        extra = f"yahoo={st.yahoo_ticker}" if st.yahoo_ticker else "yahoo=None"
        print(f"  {st.plain_symbol}: ok={st.ok} rows={st.rows} {extra}")
        if st.error:
            print(f"    error: {st.error}")
    print(f"\nLoaded frames: {len(res.data)} symbol(s)")


if __name__ == "__main__":
    main()
