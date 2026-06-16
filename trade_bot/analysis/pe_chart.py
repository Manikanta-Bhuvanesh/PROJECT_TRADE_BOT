"""
Build a compact P/E time series from Screener's ``Price-EPS`` chart API.

Screener returns sparse TTM EPS (quarterly updates) and dense daily prices.
We forward-fill the latest known EPS to each price date and compute ``PE = price / eps``.
"""
from __future__ import annotations

from bisect import bisect_right
from datetime import date
from typing import Any


def _parse_day(s: str) -> date | None:
    try:
        y, m, d = (int(x) for x in str(s).strip()[:10].split("-"))
        return date(y, m, d)
    except (TypeError, ValueError):
        return None


def _as_float(x: Any) -> float | None:
    try:
        v = float(x)
        if v > 0 and v == v:
            return v
    except (TypeError, ValueError):
        return None
    return None


def _extract_series(
    datasets: list[dict[str, Any]],
    *,
    price_substrings: tuple[str, ...] = ("price", "nse", "bse", "cmp"),
    eps_substrings: tuple[str, ...] = ("eps", "ttm"),
) -> tuple[list[tuple[date, float]], list[tuple[date, float]]]:
    """Return (price_points, eps_points) sorted by date."""
    price_pts: list[tuple[date, float]] = []
    eps_pts: list[tuple[date, float]] = []
    for ds in datasets:
        label = str(ds.get("label") or "").lower()
        values = ds.get("values") or []
        if not values:
            continue
        is_price = any(s in label for s in price_substrings)
        is_eps = any(s in label for s in eps_substrings) and not is_price
        if not is_price and not is_eps:
            continue
        bucket = price_pts if is_price else eps_pts
        for pair in values:
            if not isinstance(pair, (list, tuple)) or len(pair) < 2:
                continue
            d = _parse_day(str(pair[0]))
            px = _as_float(pair[1])
            if d is None or px is None:
                continue
            bucket.append((d, px))
    price_pts.sort(key=lambda x: x[0])
    eps_pts.sort(key=lambda x: x[0])
    return price_pts, eps_pts


def build_pe_chart_block(
    chart_json: dict[str, Any] | None,
    *,
    company_id: str,
    days: int,
    query: str = "Price-EPS",
) -> dict[str, Any]:
    """
    From raw ``/api/company/{id}/chart/?q=Price-EPS`` JSON, produce a small payload:

    - ``pe_series``: ``[{date, price, eps, pe}, ...]`` (daily where EPS known)
    - ``stats``: latest / min / max / median PE, historical percentile of latest, etc.
    """
    out: dict[str, Any] = {
        "ok": False,
        "query": query,
        "days": days,
        "company_id": company_id,
    }
    if not chart_json or not isinstance(chart_json, dict):
        out["error"] = "empty chart response"
        return out
    datasets = chart_json.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        out["error"] = "no datasets in chart response"
        return out

    price_pts, eps_pts = _extract_series(datasets)
    if not price_pts:
        out["error"] = "no price series found in chart datasets"
        return out
    if not eps_pts:
        out["error"] = "no EPS series found in chart datasets"
        return out

    eps_dates = [d for d, _ in eps_pts]
    eps_vals = [e for _, e in eps_pts]

    pe_series: list[dict[str, Any]] = []
    for d, price in price_pts:
        i = bisect_right(eps_dates, d) - 1
        if i < 0:
            continue
        eps = eps_vals[i]
        if eps <= 0:
            continue
        pe = price / eps
        if not (pe == pe) or pe <= 0:
            continue
        pe_series.append(
            {
                "date": d.isoformat(),
                "price": round(price, 4),
                "eps": round(eps, 4),
                "pe": round(pe, 4),
            }
        )

    if not pe_series:
        out["error"] = "could not align price with EPS (no valid PE points)"
        return out

    pes = [float(x["pe"]) for x in pe_series]
    latest = pe_series[-1]
    sorted_pes = sorted(pes)
    n = len(sorted_pes)
    median = sorted_pes[n // 2] if n % 2 == 1 else (sorted_pes[n // 2 - 1] + sorted_pes[n // 2]) / 2
    latest_pe = float(latest["pe"])
    inclusive_rank_pct = 100.0 * sum(1 for p in pes if p <= latest_pe) / n if n else 0.0

    half = max(1, n // 4)
    early = sum(pes[:half]) / half
    late = sum(pes[-half:]) / half
    if late > early * 1.08:
        pe_drift = "PE averaged higher at the end of the window than at the start (expanding multiple / falling EPS growth in price mix)."
    elif late < early * 0.92:
        pe_drift = "PE averaged lower at the end of the window than at the start (contracting multiple or rising EPS)."
    else:
        pe_drift = "PE level is broadly similar at the start vs end of the window (no strong multiple drift in this crude split)."

    out["ok"] = True
    out["pe_series"] = pe_series
    out["stats"] = {
        "points": n,
        "first_date": pe_series[0]["date"],
        "last_date": latest["date"],
        "latest_price": latest["price"],
        "eps_used": latest["eps"],
        "latest_pe": round(latest_pe, 4),
        "min_pe": round(min(pes), 4),
        "max_pe": round(max(pes), 4),
        "median_pe": round(median, 4),
        "latest_pe_percentile_in_window": round(inclusive_rank_pct, 1),
        "pe_drift_note": pe_drift,
    }
    return out


def pe_valuation_narrative(pe_block: dict[str, Any] | None) -> str:
    """
    Short plain-language valuation note from Price/EPS-derived PE (for chat summary only).
    Avoids counts, percentiles, and technical jargon.
    """
    if not pe_block or not pe_block.get("ok"):
        return ""
    st = pe_block.get("stats") or {}
    lo = st.get("min_pe")
    hi = st.get("max_pe")
    med = st.get("median_pe")
    cur = st.get("latest_pe")
    drift = st.get("pe_drift_note") or ""
    # soften internal drift wording
    if "expanding multiple" in drift:
        drift_plain = "Over the window, valuation multiples tended to rise more toward the end than at the start."
    elif "contracting multiple" in drift:
        drift_plain = "Over the window, multiples tended to compress toward the end compared with the start."
    else:
        drift_plain = "Multiples did not show a strong sustained move from the start to the end of the window."

    parts = [
        f"Using price and trailing EPS from Screener, the implied trailing P/E is near {cur} most recently. "
        f"Over roughly the past year shown on the site, that multiple has ranged from about {lo} to {hi}, "
        f"with a middle of the range near {med}. "
        f"{drift_plain}"
    ]
    return " ".join(parts).strip()


def pe_valuation_short(pe_block: dict[str, Any] | None) -> str:
    """One compact line for chat summaries (same data as narrative, less wording)."""
    if not pe_block or not pe_block.get("ok"):
        return ""
    st = pe_block.get("stats") or {}
    cur = st.get("latest_pe")
    lo = st.get("min_pe")
    hi = st.get("max_pe")
    med = st.get("median_pe")
    try:
        pct = float(st.get("latest_pe_percentile_in_window") or 0.0)
    except (TypeError, ValueError):
        pct = 0.0
    drift = str(st.get("pe_drift_note") or "")
    if "expanding multiple" in drift:
        drift_s = "multiple drift: up vs start of window."
    elif "contracting multiple" in drift:
        drift_s = "multiple drift: down vs start of window."
    else:
        drift_s = "multiple drift: flat vs start of window."
    return (
        f"Trailing P/E ~{cur} (window ~{lo}-{hi}, median ~{med}; "
        f"latest ~{pct:.0f}th pctile in this window). {drift_s}"
    )


def format_pe_chart_summary(pe_block: dict[str, Any] | None) -> str:
    """Short text block for Telegram from ``build_pe_chart_block`` output."""
    if not pe_block or not pe_block.get("ok"):
        err = (pe_block or {}).get("error") or "PE chart not available"
        return f"PE chart: {err}"
    st = pe_block.get("stats") or {}
    lines = [
        "PE chart (Screener Price-EPS: daily price ÷ latest TTM EPS known on that date):",
        f"  • Window: {st.get('first_date')} -> {st.get('last_date')} ({st.get('points')} trading days with EPS)",
        f"  • Latest: PE ≈ {st.get('latest_pe')} (price {st.get('latest_price')}, EPS used {st.get('eps_used')})",
        f"  • In this window: min PE {st.get('min_pe')}, median {st.get('median_pe')}, max {st.get('max_pe')}",
        f"  • Latest PE ≈ {float(st.get('latest_pe_percentile_in_window') or 0):.0f}th percentile vs own history here "
        "(higher = more expensive than more of your recent past days).",
        f"  • {st.get('pe_drift_note', '')}",
    ]
    return "\n".join(lines)
