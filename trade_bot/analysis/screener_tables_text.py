"""
Render screener.in snapshot tables as plain-text fixed-width tables (UTF-8).

Used for Telegram file attachment instead of dumping the same data as JSON.
"""
from __future__ import annotations

from typing import Any


def _cell(v: Any, max_w: int) -> str:
    s = str(v).replace("\r", " ").replace("\n", " ").strip()
    if len(s) > max_w:
        return s[: max_w - 1] + "."
    return s


def _ascii_table(columns: list[str], rows: list[dict[str, Any]], *, max_cell: int = 20) -> str:
    """Fixed-width table; truncates wide cells so lines stay readable."""
    if not columns:
        return "(empty table)\n"
    headers = [str(c) if str(c) else f"c{i}" for i, c in enumerate(columns)]
    widths = [min(max_cell, max(len(_cell(h, max_cell)), 8)) for h in headers]
    for row in rows:
        for i, col_key in enumerate(columns):
            w = len(_cell(row.get(col_key, ""), max_cell))
            widths[i] = min(max_cell, max(widths[i], w))

    def fmt_row(cells: list[str]) -> str:
        parts = [_cell(c, widths[i]).ljust(widths[i]) for i, c in enumerate(cells)]
        return "| " + " | ".join(parts) + " |\n"

    out = fmt_row(headers)
    dashes = ["-" * widths[i] for i in range(len(widths))]
    out += fmt_row(dashes)
    for row in rows:
        cells = [row.get(columns[i], "") if i < len(columns) else "" for i in range(len(headers))]
        out += fmt_row([str(c) for c in cells])
    return out


def _section_tables(sec: Any) -> list[tuple[list[str], list[dict[str, Any]]]]:
    """Return list of (columns, rows) for one HTML section (single or multiple tables)."""
    out: list[tuple[list[str], list[dict[str, Any]]]] = []
    if sec is None:
        return out
    if isinstance(sec, dict) and "rows" in sec:
        cols = [str(x) for x in (sec.get("columns") or [])]
        rows = [dict(r) for r in (sec.get("rows") or []) if isinstance(r, dict)]
        if cols or rows:
            out.append((cols, rows))
        return out
    if isinstance(sec, list):
        for part in sec:
            if not isinstance(part, dict):
                continue
            cols = [str(x) for x in (part.get("columns") or [])]
            rows = [dict(r) for r in (part.get("rows") or []) if isinstance(r, dict)]
            if cols or rows:
                out.append((cols, rows))
    return out


def _top_ratios_block(tr: dict[str, Any]) -> str:
    if not isinstance(tr, dict) or not tr:
        return ""
    cols = ["Metric", "Value"]
    rows = [{"Metric": k, "Value": v} for k, v in sorted(tr.items(), key=lambda kv: str(kv[0]))]
    return _ascii_table(cols, rows, max_cell=36)


def _peers_block(peers: list[Any]) -> str:
    if not peers:
        return "(no peers table)\n"
    keys: set[str] = set()
    for p in peers:
        if isinstance(p, dict):
            keys.update(str(k) for k in p.keys())
    columns = sorted(keys, key=str)
    if not columns:
        return "(no peers columns)\n"
    rows = [dict(r) for r in peers if isinstance(r, dict)]
    return _ascii_table(columns, rows, max_cell=18)


def _view_block(view_name: str, view: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("")
    lines.append("=" * 88)
    lines.append(f"  {view_name.upper()}")
    lines.append("=" * 88)
    lines.append(f"Company: {view.get('company_name') or '(unknown)'}")
    lines.append(f"URL path: {view.get('url_path') or ''}")
    lines.append("")

    tr = view.get("top_ratios") or {}
    if tr:
        lines.append("--- Top ratios (page header) ---")
        lines.append(_top_ratios_block(tr))
        lines.append("")

    about = view.get("about")
    if about:
        lines.append("--- About ---")
        lines.append(str(about).strip()[:4000])
        if len(str(about)) > 4000:
            lines.append("… (truncated)")
        lines.append("")

    section_titles = (
        ("quarters", "Quarterly results"),
        ("profit-loss", "Profit & loss"),
        ("balance-sheet", "Balance sheet"),
        ("cash-flow", "Cash flow"),
        ("ratios", "Ratios"),
        ("shareholding", "Shareholding"),
    )
    for key, title in section_titles:
        tables = _section_tables(view.get(key))
        if not tables:
            continue
        lines.append(f"--- {title} (`{key}`) ---")
        for ti, (cols, trows) in enumerate(tables):
            if len(tables) > 1:
                lines.append(f"(sub-table {ti + 1} of {len(tables)})")
            if not cols and not trows:
                lines.append("(empty)\n")
                continue
            lines.append(_ascii_table(cols, trows, max_cell=22))
            lines.append("")

    peers = view.get("peers") or []
    if peers:
        lines.append("--- Peers ---")
        lines.append(_peers_block(peers if isinstance(peers, list) else []))
        lines.append("")

    an = view.get("analysis") or {}
    if isinstance(an, dict):
        pros = an.get("pros") or []
        cons = an.get("cons") or []
        if pros or cons:
            lines.append("--- Analysis (pros / cons from page) ---")
            for p in pros:
                lines.append(f"+ {str(p).strip()}")
            for c in cons:
                lines.append(f"- {str(c).strip()}")
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def format_screener_tables_txt(snapshot: dict[str, Any]) -> str:
    """
    Full textual dump of tabular screener data (both views when present).

    Omits ``pe_chart`` and raw chart JSON by design.
    """
    sym = str(snapshot.get("symbol") or "").strip().upper()
    lines: list[str] = [
        f"Screener.in — {sym}",
        f"Fetched (UTC): {snapshot.get('fetched_at', '')}",
        "",
        "Notes:",
        "- Tables are copied from the site HTML; verify on screener.in.",
        "- Wide rows may be truncated for readability.",
        "",
    ]
    body: list[str] = []
    for mode in ("consolidated", "standalone"):
        v = snapshot.get(mode)
        if isinstance(v, dict) and v:
            body.append(_view_block(mode, v))

    if not body:
        lines.append("No consolidated or standalone view could be loaded.")
        errs = snapshot.get("errors") or []
        if errs:
            lines.append("")
            lines.append("Errors:")
            for e in errs:
                lines.append(f"  - {e}")
        return "\n".join(lines)

    lines.append("\n".join(body))
    errs = snapshot.get("errors") or []
    if errs:
        lines.append("")
        lines.append("--- Fetch warnings ---")
        for e in errs:
            lines.append(str(e))
    return "\n".join(lines)
