"""
Short, rule-based snapshot read from screener.in (not financial advice; verdict is not an AI).
"""
from __future__ import annotations

import statistics
from typing import Any

from trade_bot.analysis.pe_chart import pe_valuation_short
from trade_bot.analysis.screener_insights import (
    _find_row,
    _numeric_tail_from_row,
    _parse_indian_number,
    _rows_from_section,
    _table_section,
    _trend_label,
)


def _peer_column_values(peers: list[dict[str, Any]], needle: str) -> list[float]:
    out: list[float] = []
    nl = needle.lower()
    for p in peers:
        if not isinstance(p, dict):
            continue
        for k, val in p.items():
            if nl in str(k).lower():
                x = _parse_indian_number(str(val))
                if x is not None:
                    out.append(x)
                break
    return out


def _is_quarterly_cost_trend_row(label: str) -> bool:
    """Skip ratio rows and interest *income* rows when treating a line as 'expense' (up = bad)."""
    low = label.lower()
    if "ratio" in low:
        return False
    if "interest income" in low or "interest earned" in low or "income from interest" in low:
        return False
    return True


def _trend_quality_line(label: str, direction: str | None, *, higher_is_better: bool) -> str:
    """
    Map table trend (recent half vs older half) to GOOD/BAD/NEUTRAL.

    ``higher_is_better=True``: sales, profit, cash, equity, OCF.
    ``higher_is_better=False``: borrowings, operating/finance costs (rising = worse).
    """
    if direction is None:
        return f"{label}: UNCLEAR - not enough columns or no clear pattern."
    if direction == "flat":
        return f"{label}: NEUTRAL - roughly flat across the window."
    up = direction == "upward"
    if higher_is_better:
        if up:
            return f"{label}: GOOD - up in recent columns (higher is better for this line)."
        return f"{label}: BAD - down in recent columns (weaker than older periods for this line)."
    if up:
        return f"{label}: BAD - up in recent columns (more burden is worse for this line)."
    return f"{label}: GOOD - down in recent columns (less burden is better for this line)."


def _ratio_one_liner(name: str, raw: str) -> str | None:
    n = name.lower()
    v = _parse_indian_number(raw)
    if v is None:
        return f"{name}: {raw.strip()} (see Screener)"
    if "eps" in n and "p/e" not in n and "stock" not in n:
        return None
    if ("p/e" in n) or ("stock p" in n) or (" pe" in f" {n} " and "eps" not in n and "ope" not in n):
        tag = "ok" if v < 15 else ("mid" if v < 28 else "demanding")
        return f"{name}: {raw.strip()} -> P/E read: {tag}."
    if "roce" in n:
        tag = "good" if v >= 16 else ("ok" if v >= 10 else "weak")
        return f"{name}: {raw.strip()} -> ROCE: {tag}."
    if "roe" in n:
        tag = "good" if v >= 16 else ("ok" if v >= 10 else "weak")
        return f"{name}: {raw.strip()} -> ROE: {tag}."
    if "debt" in n and "equity" in n:
        tag = "comfortable" if v <= 0.8 else ("ok" if v <= 1.8 else "high risk")
        return f"{name}: {raw.strip()} -> leverage: {tag}."
    if "current ratio" in n:
        tag = "ok" if v >= 1.2 else "tight"
        return f"{name}: {raw.strip()} -> liquidity: {tag}."
    return f"{name}: {raw.strip()}"


def _peer_vs_stock(lines: list[str], peers: list[Any], top_ratios: dict[str, str]) -> None:
    lines.append("PEERS (peer table vs your headline ratios)")
    clean = [p for p in (peers or []) if isinstance(p, dict)]
    if not clean:
        lines.append("- No peer table.")
        return

    pe_vals = _peer_column_values(clean, "p/e") or _peer_column_values(clean, "pe")
    roce_vals = _peer_column_values(clean, "roce")
    mcap_vals = _peer_column_values(clean, "mar cap") or _peer_column_values(clean, "mcap")

    bits: list[str] = []
    if mcap_vals:
        bits.append(f"mcap med ~{statistics.median(mcap_vals):,.0f}")
    if pe_vals:
        bits.append(f"P/E med ~{statistics.median(pe_vals):.1f}")
    if roce_vals:
        bits.append(f"ROCE med ~{statistics.median(roce_vals):.1f}%")
    if bits:
        lines.append(f"- Peer medians: {', '.join(bits)}.")

    my_roce = next((_parse_indian_number(str(v)) for k, v in (top_ratios or {}).items() if "roce" in k.lower()), None)
    my_pe = next((_parse_indian_number(str(v)) for k, v in (top_ratios or {}).items() if "p/e" in k.lower()), None)

    better: list[str] = []
    worse: list[str] = []

    if my_roce is not None and len(roce_vals) >= 2:
        med = statistics.median(roce_vals)
        if my_roce >= med * 1.05:
            better.append(f"ROCE (you {my_roce:g}% vs peers {med:.1f}%)")
        elif my_roce <= med * 0.95:
            worse.append(f"ROCE (you {my_roce:g}% vs peers {med:.1f}%)")

    if my_pe is not None and len(pe_vals) >= 2:
        medp = statistics.median(pe_vals)
        if my_pe <= medp * 0.9:
            better.append(f"P/E (you {my_pe:g} vs peers {medp:.1f}, cheaper)")
        elif my_pe >= medp * 1.1:
            worse.append(f"P/E (you {my_pe:g} vs peers {medp:.1f}, pricier)")

    if better:
        lines.append(f"- You look BETTER on: {', '.join(better)}.")
    else:
        lines.append("- You vs peers: no clear edge on ROCE/P/E (or tied).")
    if worse:
        lines.append(f"- You look WEAKER on: {', '.join(worse)}.")


def _last_two_change(nums: list[float]) -> tuple[float | None, float | None]:
    if len(nums) < 2:
        return None, None
    a, b = nums[-2], nums[-1]
    delta = b - a
    pct = (100.0 * delta / a) if a not in (0, None) and a == a else None
    return delta, pct


def _shareholding_short(lines: list[str], view: dict[str, Any]) -> None:
    lines.append("SHAREHOLDING")
    cols, rows = _rows_from_section(_table_section(view, "shareholding"))
    if not rows or not cols:
        lines.append("- No shareholding table.")
        return
    for needles, label in (
        (("fii", "foreign institutional"), "FII"),
        (("dii", "domestic institutional"), "DII"),
        (("promoter",), "Promoter"),
        (("public",), "Public"),
    ):
        hit = _find_row(rows, cols, *needles)
        if not hit:
            continue
        nums = _numeric_tail_from_row(hit, cols, max_vals=8)
        if len(nums) < 2:
            lines.append(f"- {label}: latest ~{nums[-1]:g} (not enough columns for change).")
            continue
        d, pct = _last_two_change(nums)
        last = nums[-1]
        if d is None:
            continue
        if pct is not None:
            lines.append(f"- {label}: {last:g}% (vs prior: {d:+.2f} ppt, {pct:+.1f}% rel).")
        else:
            lines.append(f"- {label}: {last:g}% (vs prior: {d:+.2f} ppt).")


def _cfo_line(view: dict[str, Any], cfo_dir: str | None) -> str:
    cfcols, cfrows = _rows_from_section(_table_section(view, "cash-flow"))
    if not cfrows or not cfcols:
        return "Operating cash flow: no table."
    cr = _find_row(cfrows, cfcols, "cash from operating", "operating activities", "cash flows from operating")
    if not cr:
        return "Operating cash flow: row not found."
    nums = _numeric_tail_from_row(cr, cfcols, max_vals=12)
    if not nums:
        return "Operating cash flow: no numbers."
    last = nums[-1]
    trend = _trend_quality_line("OCF", cfo_dir, higher_is_better=True)
    neg = " Latest OCF column: negative." if last < 0 else " Latest OCF column: positive."
    return trend + " " + neg


def _verdict(
    *,
    rev_dir: str | None,
    prof_dir: str | None,
    cfo_dir: str | None,
    peers: list[Any],
    top_ratios: dict[str, str],
    pros: list[str],
    cons: list[str],
    pe_block: dict[str, Any] | None,
    view: dict[str, Any],
) -> tuple[str, int]:
    score = 0
    if rev_dir == "upward":
        score += 2
    elif rev_dir == "downward":
        score -= 2
    if prof_dir == "upward":
        score += 2
    elif prof_dir == "downward":
        score -= 2
    if cfo_dir == "upward":
        score += 1
    elif cfo_dir == "downward":
        score -= 1
    if len(pros) > len(cons) + 2:
        score += 1
    elif len(cons) > len(pros) + 2:
        score -= 1
    for k, v in (top_ratios or {}).items():
        if "roce" in k.lower():
            x = _parse_indian_number(str(v))
            if x is not None:
                if x >= 16:
                    score += 1
                elif x < 10:
                    score -= 1
            break
    cols_r, rows_r = _rows_from_section(_table_section(view, "ratios"))
    hit = _find_row(rows_r, cols_r, "debt to equity") if rows_r else None
    if hit:
        nums = _numeric_tail_from_row(hit, cols_r, max_vals=4)
        if nums:
            last = nums[-1]
            if last <= 0.8:
                score += 1
            elif last >= 2.5:
                score -= 1
    clean = [p for p in (peers or []) if isinstance(p, dict)]
    roce_vals = _peer_column_values(clean, "roce")
    my_roce = next((_parse_indian_number(str(v)) for k, v in (top_ratios or {}).items() if "roce" in k.lower()), None)
    if my_roce is not None and len(roce_vals) >= 3:
        med = statistics.median(roce_vals)
        if my_roce >= med * 1.08:
            score += 1
        elif my_roce <= med * 0.92:
            score -= 1
    if pe_block and pe_block.get("ok"):
        st = pe_block.get("stats") or {}
        try:
            cur, med = float(st.get("latest_pe")), float(st.get("median_pe"))
            if cur > med * 1.2:
                score -= 1
        except (TypeError, ValueError):
            pass
    if score >= 4:
        return "BUY", score
    if score <= -3:
        return "SELL", score
    return "HOLD", score


def comprehensive_screener_summary(snapshot: dict[str, Any], *, prefer_view: str = "consolidated") -> str:
    order = [prefer_view, "consolidated", "standalone"]
    view: dict[str, Any] | None = None
    used = ""
    for vn in order:
        c = snapshot.get(vn)
        if isinstance(c, dict) and (c.get("company_name") or c.get("top_ratios")):
            view, used = c, vn
            break
    if view is None:
        for vn in ("consolidated", "standalone"):
            c = snapshot.get(vn)
            if isinstance(c, dict) and c:
                view, used = c, vn
                break

    sym = str(snapshot.get("symbol") or "").strip().upper()
    lines: list[str] = [
        f"{sym} | quick scan ({used or '?'}). Not advice.",
        "",
    ]

    if view is None:
        lines.append("Load failed.")
        lines.extend(str(e) for e in (snapshot.get("errors") or []))
        return "\n".join(lines)

    name = str(view.get("company_name") or sym).strip()
    ab = (view.get("about") or "").strip()
    lines.append(f"ABOUT ({name})")
    lines.append((ab[:280] + "...") if len(ab) > 280 else (ab or "(none)"))
    lines.append("")

    an = view.get("analysis") or {}
    pros = [str(x).strip() for x in (an.get("pros") or []) if str(x).strip()]
    cons = [str(x).strip() for x in (an.get("cons") or []) if str(x).strip()]
    lines.append("PROS / CONS (Screener, full list)")
    if not pros:
        lines.append("+ (none)")
    else:
        for i, p in enumerate(pros, start=1):
            lines.append(f"+ [{i}] {p}")
    if not cons:
        lines.append("- (none)")
    else:
        for i, c in enumerate(cons, start=1):
            lines.append(f"- [{i}] {c}")
    lines.append("")

    tr = view.get("top_ratios") if isinstance(view.get("top_ratios"), dict) else {}
    lines.append("HEADLINE RATIOS")
    priority = ("p/e", "pe", "roce", "roe", "debt", "current", "dividend", "market", "book")
    keys_sorted = sorted(tr.keys(), key=lambda k: (next((i for i, p in enumerate(priority) if p in k.lower()), 99), str(k)))
    shown = 0
    for k in keys_sorted:
        line = _ratio_one_liner(str(k), str(tr[k]))
        if line:
            lines.append(f"- {line}")
            shown += 1
        if shown >= 6:
            break
    lines.append("")

    _peer_vs_stock(lines, view.get("peers") or [], tr)
    lines.append("")

    qcols, qrows = _rows_from_section(_table_section(view, "quarters"))
    rev_dir = prof_dir = exp_dir = None
    exp_row: dict[str, str] | None = None
    exp_label = ""
    if qrows and qcols:
        rr = _find_row(qrows, qcols, "sales", "revenue", "total income")
        if rr:
            rev_dir = _trend_label(_numeric_tail_from_row(rr, qcols, max_vals=20))
        pr = _find_row(qrows, qcols, "net profit", "pat", "profit for the period")
        if pr:
            prof_dir = _trend_label(_numeric_tail_from_row(pr, qcols, max_vals=20))
        exp_row = _find_row(
            qrows,
            qcols,
            "operating expenses",
            "total expenses",
            "other expenses",
            "finance cost",
            "finance costs",
            "interest expense",
            "interest paid",
            "borrowing cost",
            "cost of material",
            "cost of raw material",
        )
        if exp_row:
            exp_label = (exp_row.get(qcols[0]) or "Expenses / finance costs").strip()
            if _is_quarterly_cost_trend_row(exp_label):
                exp_dir = _trend_label(_numeric_tail_from_row(exp_row, qcols, max_vals=20))
            else:
                exp_row = None
                exp_label = ""
    if prof_dir is None:
        plcols, plrows = _rows_from_section(_table_section(view, "profit-loss"))
        if plrows and plcols:
            pr2 = _find_row(plrows, plcols, "net profit", "pat")
            if pr2:
                prof_dir = _trend_label(_numeric_tail_from_row(pr2, plcols, max_vals=16))

    cfo_dir = None
    cfcols, cfrows = _rows_from_section(_table_section(view, "cash-flow"))
    if cfrows and cfcols:
        cr = _find_row(cfrows, cfcols, "cash from operating", "operating activities", "cash flows from operating")
        if cr:
            cfo_dir = _trend_label(_numeric_tail_from_row(cr, cfcols, max_vals=16))

    lines.append("QUARTERLY (trend = recent columns vs older; meaning depends on line)")
    if qrows and qcols:
        rr = _find_row(qrows, qcols, "sales", "revenue", "total income")
        lab = "Sales/Revenue"
        if rr and qcols:
            lab = (rr.get(qcols[0]) or lab).strip() or lab
        lines.append(f"- {_trend_quality_line(lab, rev_dir, higher_is_better=True)}")
        if exp_row and exp_label:
            lines.append(f"- {_trend_quality_line(exp_label, exp_dir, higher_is_better=False)}")
        pr = _find_row(qrows, qcols, "net profit", "pat", "profit for the period")
        labp = "Net profit"
        if pr and qcols:
            labp = (pr.get(qcols[0]) or labp).strip() or labp
        lines.append(f"- {_trend_quality_line(labp, prof_dir, higher_is_better=True)}")
    else:
        lines.append("- No quarterly table.")

    lines.append("")
    lines.append("CASH FLOW")
    lines.append(f"- {_cfo_line(view, cfo_dir)}")

    lines.append("")
    lines.append("BALANCE SHEET (high level)")
    bcols, brows = _rows_from_section(_table_section(view, "balance-sheet"))
    if brows and bcols:
        for needles, short, higher_ok in (
            (
                (
                    "borrowings",
                    "long term borrowings",
                    "short term borrowings",
                    "long-term borrowings",
                    "short-term borrowings",
                    "total borrowings",
                    "net debt",
                ),
                "Debt/borrowings",
                False,
            ),
            (("cash", "cash and bank", "bank balances"), "Cash", True),
            (("equity", "net worth", "shareholders fund"), "Equity/Net worth", True),
        ):
            hit = _find_row(brows, bcols, *needles)
            if hit:
                nums = _numeric_tail_from_row(hit, bcols, max_vals=8)
                if len(nums) >= 2:
                    td = _trend_label(nums)
                    lines.append(
                        f"- {_trend_quality_line(short, td, higher_is_better=higher_ok)} Latest ~{nums[-1]:g}."
                    )
                elif nums:
                    lines.append(f"- {short}: latest ~{nums[-1]:g}.")
    else:
        lines.append("- No balance sheet table.")

    lines.append("")
    _shareholding_short(lines, view)

    lines.append("")
    lines.append("P/E (Price / TTM EPS)")
    pe = pe_valuation_short(snapshot.get("pe_chart"))
    lines.append(pe or "- Not available.")

    v, sc = _verdict(
        rev_dir=rev_dir,
        prof_dir=prof_dir,
        cfo_dir=cfo_dir,
        peers=view.get("peers") or [],
        top_ratios=tr,
        pros=pros,
        cons=cons,
        pe_block=snapshot.get("pe_chart"),
        view=view,
    )
    lines.append("")
    lines.append(f"VERDICT: **{v}** (score {sc}, rules-only). Not advice.")

    return "\n".join(lines)
