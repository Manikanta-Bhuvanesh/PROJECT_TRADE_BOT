from __future__ import annotations

import re
from typing import Any

from trade_bot.analysis.pe_chart import pe_valuation_narrative


def _parse_indian_number(cell: str) -> float | None:
    """Parse screener table cells: commas, negatives in parentheses, trailing %."""
    if cell is None:
        return None
    s = str(cell).strip()
    if not s or s in ("-", "—"):
        return None
    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg = True
        s = s[1:-1].strip()
    s = s.replace(",", "")
    low = s.lower()
    if low.endswith(" cr") or low.endswith("cr"):
        s = re.sub(r"(?i)\s*cr$", "", s).strip()
    elif re.search(r"(?i)lacs?$", low):
        s = re.sub(r"(?i)\s*lacs?$", "", s).strip()
    s = re.sub(r"(?i)\s*%$", "", s).strip()
    try:
        v = float(s)
        return -v if neg else v
    except ValueError:
        return None


def _table_section(view: dict[str, Any] | None, key: str) -> dict[str, Any] | list | None:
    if not view or not isinstance(view, dict):
        return None
    return view.get(key)


def _rows_from_section(sec: dict[str, Any] | list | None) -> tuple[list[str], list[dict[str, str]]]:
    if sec is None:
        return [], []
    if isinstance(sec, list):
        rows: list[dict[str, str]] = []
        cols: list[str] = []
        for part in sec:
            if not isinstance(part, dict):
                continue
            c = part.get("columns") or []
            for r in part.get("rows") or []:
                if isinstance(r, dict):
                    rows.append({str(k): str(v) for k, v in r.items()})
            if not cols and c:
                cols = [str(x) for x in c]
        return cols, rows
    if isinstance(sec, dict) and "rows" in sec:
        cols = [str(x) for x in (sec.get("columns") or [])]
        rows = [{str(k): str(v) for k, v in r.items()} for r in (sec.get("rows") or []) if isinstance(r, dict)]
        return cols, rows
    return [], []


def _find_row(rows: list[dict[str, str]], columns: list[str], *needles: str) -> dict[str, str] | None:
    nl = [n.lower() for n in needles]
    for row in rows:
        label = ""
        if columns:
            label = (row.get(columns[0]) or "").lower()
        if not label:
            label = " ".join(str(v).lower() for v in row.values())
        if any(n in label for n in nl):
            return row
    return None


def _numeric_tail_from_row(row: dict[str, str], columns: list[str], *, max_vals: int = 12) -> list[float]:
    if not columns:
        return []
    nums: list[float] = []
    for col in columns[1:]:
        v = _parse_indian_number(row.get(col, ""))
        if v is not None:
            nums.append(v)
    return nums[-max_vals:]


def _trend_label(vals: list[float]) -> str | None:
    if len(vals) < 4:
        return None
    mid = len(vals) // 2
    a = vals[:mid]
    b = vals[mid:]
    if not a or not b:
        return None
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    if mb > ma * 1.05:
        return "upward"
    if mb < ma * 0.95:
        return "downward"
    return "flat"


def _trend_bucket(vals: list[float]) -> str | None:
    return _trend_label(vals)


def _trend_sentence(label: str, direction: str | None) -> str:
    lab = (label or "This line").strip()
    if direction == "upward":
        return f"{lab} improves in more recent periods than in older columns in the table."
    if direction == "downward":
        return f"{lab} is softer in more recent periods than in older columns in the table."
    if direction == "flat":
        return f"{lab} is broadly stable across the periods shown."
    return f"{lab}: the table does not show a clear up-or-down pattern across periods."


def heuristic_fundamental_summary(snapshot: dict[str, Any], *, prefer_view: str = "consolidated") -> str:
    """
    Short narrative read of fundamentals (no row counts, no 'parsed', no JSON references).

    Tables themselves are in the separate ``.txt`` attachment from ``format_screener_tables_txt``.
    """
    sym = str(snapshot.get("symbol") or "").strip().upper()
    paragraphs: list[str] = []

    pick_order: list[str] = []
    if prefer_view in ("consolidated", "standalone"):
        pick_order.append(prefer_view)
    for vn in ("consolidated", "standalone"):
        if vn not in pick_order:
            pick_order.append(vn)

    view: dict[str, Any] | None = None
    used = "none"
    for vname in pick_order:
        cand = snapshot.get(vname)
        if isinstance(cand, dict) and (cand.get("company_name") or cand.get("top_ratios")):
            view = cand
            used = vname
            break
    if view is None:
        for vname in ("consolidated", "standalone"):
            cand = snapshot.get(vname)
            if isinstance(cand, dict) and cand:
                view = cand
                used = vname
                break

    if view is None:
        out = [f"{sym}: company page could not be read."]
        errs = snapshot.get("errors") or []
        if errs:
            out.append("Fetch issues: " + "; ".join(str(e) for e in errs))
        return " ".join(out)

    name = view.get("company_name") or sym
    paragraphs.append(
        f"{name} ({used} view on screener.in). This is a quick machine-assisted read, not advice. "
        f"Use the attached text file for full tables."
    )

    pe_txt = pe_valuation_narrative(snapshot.get("pe_chart"))
    if pe_txt:
        paragraphs.append(pe_txt)

    tr = view.get("top_ratios") or {}
    if isinstance(tr, dict) and tr:
        bits = [f"{k} is shown as {v}." for k, v in sorted(tr.items(), key=lambda kv: str(kv[0]))[:12]]
        paragraphs.append(" ".join(bits))

    rev_bucket: str | None = None
    prof_bucket: str | None = None

    for sec_name, needles in (
        ("quarters", ("sales", "revenue", "total income")),
        ("profit-loss", ("net profit", "pat", "profit for the period")),
    ):
        sec = _table_section(view, sec_name)
        cols, rows = _rows_from_section(sec)
        row = _find_row(rows, cols, *needles) if rows else None
        if row and cols:
            nums = _numeric_tail_from_row(row, cols)
            direction = _trend_label(nums)
            label = (row.get(cols[0]) if cols else "").strip() or ("Revenue" if sec_name == "quarters" else "Profit")
            if sec_name == "quarters":
                rev_bucket = _trend_bucket(nums)
            else:
                prof_bucket = _trend_bucket(nums)
            paragraphs.append(_trend_sentence(label, direction))

    bs = _table_section(view, "balance-sheet")
    cols_b, rows_b = _rows_from_section(bs)
    if rows_b and cols_b:
        for needles, friendly in (
            (("borrowings", "debt", "long term borrowings"), "Borrowings"),
            (("total equity", "shareholder", "equity"), "Equity"),
            (("cash and bank", "cash equivalents", "cash"), "Cash and equivalents"),
        ):
            hit = _find_row(rows_b, cols_b, *needles)
            if hit:
                nums = _numeric_tail_from_row(hit, cols_b, max_vals=8)
                direction = _trend_label(nums)
                label = (hit.get(cols_b[0]) if cols_b else "").strip() or friendly
                paragraphs.append(_trend_sentence(label, direction))

    cf = _table_section(view, "cash-flow")
    cols_c, rows_c = _rows_from_section(cf)
    if rows_c and cols_c:
        hit = _find_row(
            rows_c,
            cols_c,
            "cash from operating",
            "cash flows from operating",
            "operating activities",
        )
        if hit:
            nums = _numeric_tail_from_row(hit, cols_c, max_vals=10)
            direction = _trend_label(nums)
            label = (hit.get(cols_c[0]) if cols_c else "").strip() or "Cash from operations"
            paragraphs.append(_trend_sentence(label, direction))

    ratios_sec = _table_section(view, "ratios")
    cols_r, rows_r = _rows_from_section(ratios_sec)
    if rows_r and cols_r:
        ratio_bits: list[str] = []
        seen: set[str] = set()
        for metric_needle, friendly in (
            ("roe", "Return on equity"),
            ("return on equity", "Return on equity"),
            ("return on capital employed", "Return on capital employed"),
            ("roce", "Return on capital employed"),
            ("opm", "Operating margin"),
            ("net profit margin", "Net profit margin"),
            ("debt to equity", "Debt to equity"),
            ("current ratio", "Current ratio"),
            ("interest coverage", "Interest coverage"),
        ):
            if friendly in seen:
                continue
            hit = _find_row(rows_r, cols_r, metric_needle)
            if hit:
                nums = _numeric_tail_from_row(hit, cols_r, max_vals=6)
                last = nums[-1] if nums else None
                if last is not None:
                    ratio_bits.append(f"{friendly} in the latest column shown is about {last:g}.")
                    seen.add(friendly)
        if ratio_bits:
            paragraphs.append(" ".join(ratio_bits))

    sh = _table_section(view, "shareholding")
    cols_s, rows_s = _rows_from_section(sh)
    if rows_s and cols_s:
        sh_bits: list[str] = []
        for needles, friendly in (
            (("promoter", "promoters"), "Promoters"),
            (("fii", "foreign institutional"), "FIIs"),
            (("dii", "domestic institutional"), "DIIs"),
            (("public",), "Public"),
        ):
            hit = _find_row(rows_s, cols_s, *needles)
            if hit:
                nums = _numeric_tail_from_row(hit, cols_s, max_vals=4)
                last = nums[-1] if nums else None
                if last is not None:
                    sh_bits.append(f"{friendly} are near {last:g} in the latest column.")
        if sh_bits:
            paragraphs.append(" ".join(sh_bits))

    an = view.get("analysis") or {}
    if isinstance(an, dict):
        pros = [str(x).strip() for x in (an.get("pros") or []) if str(x).strip()]
        cons = [str(x).strip() for x in (an.get("cons") or []) if str(x).strip()]
        if pros:
            paragraphs.append("Strengths the page highlights include: " + "; ".join(pros[:4]) + ".")
        if cons:
            paragraphs.append("Risks or drawbacks the page lists include: " + "; ".join(cons[:4]) + ".")

    if rev_bucket and prof_bucket:
        if rev_bucket == "upward" and prof_bucket == "downward":
            paragraphs.append(
                "Big picture: revenue looks stronger in recent columns than profit, "
                "which can mean pressure on margins or one-off costs worth checking in the tables."
            )
        elif rev_bucket == "downward" and prof_bucket == "downward":
            paragraphs.append(
                "Big picture: both revenue and profit look softer in recent columns than earlier ones."
            )
        elif rev_bucket == "upward" and prof_bucket == "upward":
            paragraphs.append(
                "Big picture: both revenue and profit look better in recent columns than earlier ones."
            )
        elif rev_bucket == "flat" and prof_bucket == "flat":
            paragraphs.append("Big picture: revenue and profit look broadly steady across the periods shown.")
        else:
            paragraphs.append(
                "Big picture: revenue and profit do not move in the same direction across the table; "
                "worth reading the quarterly and P&L blocks in the attachment."
            )

    paragraphs.append("Confirm every number on screener.in before acting on it.")

    return "\n\n".join(paragraphs)
