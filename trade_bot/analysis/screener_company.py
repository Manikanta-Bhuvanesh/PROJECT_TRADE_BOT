"""
Fetch fundamental data from `screener.in` for an NSE/BSE-style symbol.

HAR analysis (``screener.har``, RVNL)
------------------------------------
- The browser loads the company **HTML** document (standalone default URL
  ``/company/RVNL/`` redirects canonically to consolidated; the **numbers** for
  standalone vs consolidated still differ: standalone uses the ``data-warehouse-id``
  from the default page, consolidated uses another warehouse row tied to
  ``data-consolidated=\"true\"`` on ``#company-info``).
- Dynamic fragments match:
  ``GET /api/company/{warehouse_id}/peers/`` (HTML table of peers — warehouse id
  from ``#company-info``) and
  ``GET /api/company/{company_id}/chart/?q=Price-DMA50-DMA200-Volume&days=365``
  (JSON price + DMAs + volume — ``data-company-id`` from ``#company-info``).

This module mirrors that behaviour: it fetches **both** ``…/consolidated/`` and the
default ``…/{SYM}/`` page, parses the same sections/tables the UI shows, pulls peers
for each warehouse, and pulls chart JSON once when both views share the same
``data-company-id``.

Legacy open-source frontend (archived 2019)
--------------------------------------------
The old React client at `Mittal-Analytics/Screener.in` on GitHub
(https://github.com/Mittal-Analytics/Screener.in) is **not** the live site today, but
it documents naming that still lines up with current behaviour:

- ``app/api.js`` — REST base ``/api/``, trailing slashes, ``Api.cid(id, component)``
  → ``/api/company/{id}/{component}/``.
- ``app/company/peers.jsx`` — peers XHR uses **warehouse** id
  (``company.warehouse_set.id``), optional ``industry`` query param.
- ``app/company/pricechart.jsx`` — historically loaded **prices** via
  ``/api/company/{cid}/prices/``; the modern UI uses the **chart** JSON endpoint
  instead (this module follows the current ``/chart/`` API).

**Disclaimer:** Unofficial helper for personal research. Respect Screener.in rate
limits and terms of use. Not financial advice.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Any, Literal

from bs4 import BeautifulSoup

from trade_bot.analysis.pe_chart import build_pe_chart_block

SCREENER_BASE = "https://www.screener.in"
# Public reference: https://github.com/Mittal-Analytics/Screener.in (archived React client)
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/125.0.0.0 Safari/537.36"
)

# Sections observed on company pages (see HAR / live HTML). ``insights`` omitted — not loaded.
_SECTION_IDS: tuple[str, ...] = (
    "quarters",
    "profit-loss",
    "balance-sheet",
    "cash-flow",
    "ratios",
    "shareholding",
)

# Chart API: only Price + TTM EPS; PE is derived in ``pe_chart.build_pe_chart_block``.
_PE_CHART_QUERY = "Price-EPS"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sleep_polite(seconds: float) -> None:
    if seconds > 0:
        time.sleep(seconds)


def _http_get(url: str, *, timeout: int = 45) -> bytes:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": DEFAULT_USER_AGENT,
            "Accept": "*/*",
            "Accept-Language": "en-US,en;q=0.9",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _parse_company_info_tag(html: str) -> dict[str, Any]:
    m = re.search(
        r'<[^>]*\bid="company-info"[^>]*>',
        html,
        re.IGNORECASE | re.DOTALL,
    )
    if not m:
        return {}
    tag = m.group(0)
    out: dict[str, Any] = {}
    mid = re.search(r'data-company-id="(\d+)"', tag, re.I)
    if mid:
        out["company_id"] = mid.group(1)
    wid = re.search(r'data-warehouse-id="(\d+)"', tag, re.I)
    if wid:
        out["warehouse_id"] = wid.group(1)
    cons = re.search(r'data-consolidated="(true|false)"', tag, re.I)
    if cons:
        out["is_consolidated_flag"] = cons.group(1) == "true"
    return out


def _canonical_path(html: str) -> str | None:
    m = re.search(r'<link[^>]+rel="canonical"[^>]+href="([^"]+)"', html, re.I)
    return m.group(1) if m else None


def _page_title(html: str) -> str | None:
    m = re.search(r"<title>([^<]+)</title>", html, re.I)
    return m.group(1).strip() if m else None


def _company_display_name(soup: BeautifulSoup) -> str | None:
    h1 = soup.find("h1", class_=lambda c: c and "shrink-text" in str(c))
    if h1:
        return h1.get_text(" ", strip=True) or None
    h1b = soup.find("h1")
    return h1b.get_text(" ", strip=True) if h1b else None


def _parse_top_ratios(soup: BeautifulSoup) -> dict[str, str]:
    ul = soup.find("ul", id="top-ratios")
    if not ul:
        return {}
    out: dict[str, str] = {}
    for li in ul.find_all("li", recursive=False):
        name_el = li.find("span", class_="name")
        val_el = li.find("span", class_=lambda c: c and "value" in str(c))
        if not name_el or not val_el:
            continue
        key = name_el.get_text(" ", strip=True)
        val = val_el.get_text(" ", strip=True)
        if key:
            out[key] = val
    return out


def _table_to_rows(table) -> dict[str, Any]:
    """First table row = column headers; following rows = records."""
    rows_el = table.find_all("tr")
    if not rows_el:
        return {"columns": [], "rows": []}
    header_cells = [
        c.get_text(" ", strip=True) for c in rows_el[0].find_all(["th", "td"])
    ]
    headers = [h if h else "metric" for h in header_cells]
    body: list[dict[str, str]] = []
    for tr in rows_el[1:]:
        cells = [c.get_text(" ", strip=True) for c in tr.find_all(["th", "td"])]
        if not any(cells):
            continue
        if len(cells) < len(headers):
            cells = cells + [""] * (len(headers) - len(cells))
        elif len(cells) > len(headers):
            cells = cells[: len(headers)]
        body.append({headers[i]: cells[i] for i in range(len(headers))})
    return {"columns": headers, "rows": body}


def _parse_section_tables(soup: BeautifulSoup, section_id: str) -> list[dict[str, Any]]:
    sec = soup.find("section", id=section_id)
    if not sec:
        return []
    tables = sec.find_all("table", class_=lambda c: c and "data-table" in str(c))
    parsed: list[dict[str, Any]] = []
    for i, tbl in enumerate(tables):
        one = _table_to_rows(tbl)
        one["table_index"] = i
        parsed.append(one)
    return parsed


def _parse_analysis(soup: BeautifulSoup) -> dict[str, list[str]]:
    sec = soup.find("section", id="analysis")
    if not sec:
        return {"pros": [], "cons": []}
    pros = [
        li.get_text(" ", strip=True)
        for li in sec.select(".pros ul li")
        if li.get_text(strip=True)
    ]
    cons = [
        li.get_text(" ", strip=True)
        for li in sec.select(".cons ul li")
        if li.get_text(strip=True)
    ]
    return {"pros": pros, "cons": cons}


def _parse_documents(soup: BeautifulSoup) -> dict[str, list[dict[str, str]]]:
    sec = soup.find("section", id="documents")
    if not sec:
        return {}
    out: dict[str, list[dict[str, str]]] = {}
    for block in sec.select("div.documents"):
        h3 = block.find("h3")
        title = h3.get_text(" ", strip=True) if h3 else "misc"
        items: list[dict[str, str]] = []
        for li in block.select("ul.list-links li"):
            a = li.find("a", href=True)
            if not a:
                continue
            desc = li.find("div", class_=lambda c: c and "smaller" in str(c))
            items.append(
                {
                    "title": a.get_text(" ", strip=True),
                    "url": a["href"],
                    "note": desc.get_text(" ", strip=True) if desc else "",
                }
            )
        if items:
            out[title] = items
    return out


def _parse_about(soup: BeautifulSoup) -> str | None:
    box = soup.select_one("div.show-more-box.about")
    if not box:
        return None
    t = box.get_text(" ", strip=True)
    return t or None


def _parse_peers_html(html: str) -> list[dict[str, Any]]:
    soup = BeautifulSoup(html, "html.parser")
    tbl = soup.find("table", class_=lambda c: c and "data-table" in str(c))
    if not tbl:
        return []
    rec = _table_to_rows(tbl)
    return rec["rows"]


def _load_chart_json(company_id: str, *, days: int, query: str = _PE_CHART_QUERY) -> dict[str, Any] | None:
    q = urllib.parse.quote(query, safe="")
    url = f"{SCREENER_BASE}/api/company/{company_id}/chart/?q={q}&days={days}"
    try:
        raw = _http_get(url)
        return json.loads(raw.decode("utf-8"))
    except (urllib.error.HTTPError, json.JSONDecodeError, UnicodeDecodeError):
        return None


def _load_peers_rows(warehouse_id: str) -> list[dict[str, Any]]:
    # Peers HTML: same path as legacy app/company/peers.jsx (warehouse_set.id, not company pk).
    url = f"{SCREENER_BASE}/api/company/{warehouse_id}/peers/"
    try:
        raw = _http_get(url).decode("utf-8", "replace")
        return _parse_peers_html(raw)
    except urllib.error.HTTPError:
        return []


def _fetch_html(path: str) -> str:
    url = f"{SCREENER_BASE}{path}"
    return _http_get(url).decode("utf-8", "replace")


def _parse_single_view(
    *,
    soup: BeautifulSoup,
    html: str,
    url_path: str,
    warehouse_id: str | None,
    company_id: str | None,
    include_documents: bool = False,
) -> dict[str, Any]:
    block: dict[str, Any] = {
        "url_path": url_path,
        "page_title": _page_title(html),
        "company_name": _company_display_name(soup),
        "canonical_path": _canonical_path(html),
        "company_info": _parse_company_info_tag(html),
        "about": _parse_about(soup),
        "top_ratios": _parse_top_ratios(soup),
        "analysis": _parse_analysis(soup),
    }
    if include_documents:
        block["documents"] = _parse_documents(soup)
    for sid in _SECTION_IDS:
        tables = _parse_section_tables(soup, sid)
        if not tables:
            block[sid] = {"columns": [], "rows": []}
        elif len(tables) == 1:
            block[sid] = tables[0]
        else:
            block[sid] = tables

    if warehouse_id:
        block["peers"] = _load_peers_rows(warehouse_id)
    else:
        block["peers"] = []
    return block


def fetch_screener_snapshot(
    symbol: str,
    *,
    include_pe_chart: bool = True,
    chart_days: int = 365,
    request_delay_s: float = 0.35,
    include_documents: bool = False,
) -> dict[str, Any]:
    """
    Pull consolidated + standalone company data from screener.in.

    Parameters
    ----------
    symbol:
        Screener slug (usually the NSE symbol), e.g. ``RELIANCE``, ``RVNL``.
    include_pe_chart:
        When True, fetches ``/api/company/{id}/chart/?q=Price-EPS`` once and stores a slim
        ``pe_chart`` block (daily PE from price ÷ latest known TTM EPS). No DMA/volume chart.
    chart_days:
        ``days`` query parameter for the chart API (default 365).
    request_delay_s:
        Small pause between HTTP calls to reduce load / throttling risk.
    include_documents:
        When True, parses the ``documents`` section (PDF links). Default False so payloads
        stay focused on financial tables.

    Returns
    -------
    dict
        ``symbol``, ``fetched_at``, ``consolidated``, ``standalone``, ``pe_chart`` (or None),
        ``notes`` / ``errors``. The ``insights`` HTML section is not parsed.
    """
    sym = symbol.strip().upper()
    if not sym:
        raise ValueError("symbol must be non-empty")

    result: dict[str, Any] = {
        "symbol": sym,
        "fetched_at": _now_iso(),
        "consolidated": None,
        "standalone": None,
        "pe_chart": None,
        "errors": [],
    }

    paths: dict[Literal["consolidated", "standalone"], str] = {
        "consolidated": f"/company/{sym}/consolidated/",
        "standalone": f"/company/{sym}/",
    }

    html_cache: dict[str, str] = {}
    for mode, path in paths.items():
        try:
            html_cache[mode] = _fetch_html(path)
        except urllib.error.HTTPError as e:
            result["errors"].append({"view": mode, "error": f"HTTP {e.code}", "path": path})
            html_cache[mode] = ""
        except Exception as e:  # noqa: BLE001 — surface network/parsing issues
            result["errors"].append({"view": mode, "error": str(e), "path": path})
            html_cache[mode] = ""
        _sleep_polite(request_delay_s)

    for mode in ("consolidated", "standalone"):
        html = html_cache.get(mode, "")
        if not html:
            continue
        soup = BeautifulSoup(html, "html.parser")
        info = _parse_company_info_tag(html)
        wid = info.get("warehouse_id")
        cid = info.get("company_id")
        view = _parse_single_view(
            soup=soup,
            html=html,
            url_path=paths[mode],
            warehouse_id=str(wid) if wid else None,
            company_id=str(cid) if cid else None,
            include_documents=include_documents,
        )
        result[mode] = view
        _sleep_polite(request_delay_s)

    company_id: str | None = None
    for mode in ("consolidated", "standalone"):
        v = result.get(mode)
        if isinstance(v, dict):
            c = (v.get("company_info") or {}).get("company_id")
            if c:
                company_id = str(c)
                break

    if include_pe_chart and company_id:
        raw = _load_chart_json(company_id, days=chart_days, query=_PE_CHART_QUERY)
        _sleep_polite(request_delay_s)
        result["pe_chart"] = build_pe_chart_block(
            raw,
            company_id=company_id,
            days=chart_days,
            query=_PE_CHART_QUERY,
        )

    # If only one view exists, note it.
    if result["consolidated"] is None and result["standalone"] is None:
        result["notes"] = "No data loaded; check symbol or HTTP errors."
    else:
        result["notes"] = (
            "consolidated = /consolidated/ ; standalone = default company URL. "
            "``pe_chart`` = one Price-EPS API call; PE = price / latest TTM EPS known on each day. "
            "No insights section or DMA/volume chart payloads."
        )
    return result


def format_screener_snapshot_text(snapshot: dict[str, Any]) -> str:
    """Human-readable summary for logging / CLI."""
    lines: list[str] = []
    lines.append(f"Symbol: {snapshot.get('symbol')}")
    lines.append(f"Fetched: {snapshot.get('fetched_at')}")
    for mode in ("consolidated", "standalone"):
        v = snapshot.get(mode)
        lines.append(f"\n--- {mode.upper()} ---")
        if not v:
            lines.append("  (missing)")
            continue
        lines.append(f"  Name: {v.get('company_name')}")
        tr = v.get("top_ratios") or {}
        lines.append(f"  Top ratios: {len(tr)} items")
        for sid in _SECTION_IDS:
            sec = v.get(sid)
            if isinstance(sec, dict) and "rows" in sec:
                nrows = len(sec["rows"])
                lines.append(f"  {sid}: 1 table, {nrows} rows")
            elif isinstance(sec, list):
                nrows = sum(
                    len(x.get("rows", []))
                    for x in sec
                    if isinstance(x, dict)
                )
                lines.append(f"  {sid}: {len(sec)} tables, {nrows} rows total")
            else:
                lines.append(f"  {sid}: (empty)")
        peers = v.get("peers") or []
        lines.append(f"  peers: {len(peers)} rows")
        if "documents" in v:
            doc = v.get("documents") or {}
            nd = sum(len(vv) for vv in doc.values()) if isinstance(doc, dict) else 0
            lines.append(f"  document links: {nd}")
    pe = snapshot.get("pe_chart")
    lines.append("\n--- PE_CHART (Price-EPS derived) ---")
    if isinstance(pe, dict) and pe.get("ok"):
        st = pe.get("stats") or {}
        lines.append(
            f"  points={st.get('points')} latest_pe={st.get('latest_pe')} "
            f"range=[{st.get('min_pe')}, {st.get('max_pe')}]"
        )
    elif isinstance(pe, dict):
        lines.append(f"  failed: {pe.get('error', 'unknown')}")
    else:
        lines.append("  (not loaded)")
    if snapshot.get("errors"):
        lines.append("\nErrors:")
        for e in snapshot["errors"]:
            lines.append(f"  {e}")
    return "\n".join(lines)


def snapshot_to_json(snapshot: dict[str, Any], *, indent: int = 2) -> str:
    """Serialize a snapshot to a UTF-8 JSON string (suitable for writing to disk)."""
    return json.dumps(snapshot, indent=indent, ensure_ascii=False)


if __name__ == "__main__":
    import sys

    syms = sys.argv[1:] or ["RVNL", "RELIANCE", "TCS", "INFY", "HDFCBANK"]
    for s in syms:
        snap = fetch_screener_snapshot(s, request_delay_s=0.4)
        print(format_screener_snapshot_text(snap))
        print()
