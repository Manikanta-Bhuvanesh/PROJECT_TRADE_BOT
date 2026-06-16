from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def _normalize_signal(val: Any) -> str:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return "hold"
    s = str(val).strip().lower()
    if s in ("buy", "b", "long"):
        return "buy"
    if s in ("sell", "s", "short_exit", "exit"):
        return "sell"
    return "hold"


def _normalize_signal_intraday(val: Any) -> str:
    """Canonical intraday tokens: long_buy / long_sell / short_sell / short_buy / hold."""
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return "hold"
    s = str(val).strip().lower().replace(" ", "_").replace("-", "_")
    if s in ("buy", "b", "long", "long_buy", "lb"):
        return "long_buy"
    if s in ("sell", "s", "long_sell", "exit", "close_long"):
        return "long_sell"
    if s in ("short_sell", "short_entry", "ss"):
        return "short_sell"
    if s in ("short_buy", "short_cover", "cover", "buy_to_cover", "btc"):
        return "short_buy"
    return "hold"


def _find_columns(df: pd.DataFrame, date_col: str, close_col: str, signal_col: str) -> tuple[str, str, str]:
    lower_map = {c.lower(): c for c in df.columns}
    for want in (date_col, close_col, signal_col):
        if want not in df.columns and want.lower() in lower_map:
            df.rename(columns={lower_map[want.lower()]: want}, inplace=True)
    missing = [c for c in (date_col, close_col, signal_col) if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}. Present: {list(df.columns)}")
    return date_col, close_col, signal_col


def _median_bar_days(dates: pd.Series) -> float:
    if len(dates) < 2:
        return 1.0
    dt = pd.to_datetime(dates, errors="coerce")
    gaps = dt.diff().dt.total_seconds().div(86400.0).dropna()
    if gaps.empty:
        return 1.0
    return float(gaps.median())


def _periods_per_year(dates: pd.Series) -> float:
    med = _median_bar_days(dates)
    if med <= 1.5:
        return 252.0
    if med <= 8.0:
        return 52.0
    if med <= 35.0:
        return 12.0
    return max(1.0, 365.25 / med)


def _annualized_return_pct(final_equity: float, initial_capital: float, span_days: int) -> float:
    """
    Compound annualized return (%).

    ``(equity/capital) ** (365.25/days) - 1`` is undefined in reals when the ratio is
    negative (Python would return a **complex** for a fractional exponent).
    """
    if initial_capital <= 0 or span_days <= 0:
        return 0.0
    ratio = float(final_equity) / float(initial_capital)
    if not math.isfinite(ratio) or ratio <= 0:
        return float("nan")
    exp = 365.25 / float(span_days)
    try:
        powered = ratio**exp
    except (OverflowError, OSError, ValueError):
        return float("nan")
    if isinstance(powered, complex):
        return float("nan")
    out = (powered - 1.0) * 100.0
    if isinstance(out, complex):
        return float("nan")
    outf = float(out)
    if not math.isfinite(outf):
        return float("nan")
    return outf


def analyze_trades(
    df: pd.DataFrame,
    stop_loss_pct: float | None = None,
    *,
    initial_capital: float = 100_0000.0,
    date_col: str = "date",
    close_col: str = "close",
    signal_col: str = "signal",
    allow_short: bool = False,
) -> dict[str, Any]:
    """
    Analyse completed round-trips from price signals and ``close``.

    When ``allow_short`` is ``False`` (default), signals are **long-only**:
    ``buy`` / ``sell`` / ``hold`` (aliases: b, s, long, exit).

    When ``allow_short`` is ``True`` (intraday packages), signals are:
    ``long_buy``, ``long_sell``, ``short_sell``, ``short_buy``, ``hold``.
    ``buy`` / ``sell`` are treated as ``long_buy`` / ``long_sell`` for convenience.

    stop_loss_pct
        If set (e.g. ``5`` for 5%%), a **long** is closed when ``close`` falls to
        ``entry_price * (1 - stop_loss_pct/100)`` or below; a **short** is closed
        when ``close`` rises to ``entry_price * (1 + stop_loss_pct/100)`` or above.
        Evaluated before a closing signal on the same bar. If ``None``, exits are
        only from exit signals or end-of-data.
    initial_capital
        Starting portfolio value for equity compounding and return metrics.
    date_col, close_col, signal_col
        Column names (case-insensitive fallback: renames to these canonical names).
    """
    if df is None or df.empty:
        return _empty_result(initial_capital, allow_short=allow_short)

    work = df.copy()
    _find_columns(work, date_col, close_col, signal_col)

    work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
    work[close_col] = pd.to_numeric(work[close_col], errors="coerce")
    work.dropna(subset=[date_col, close_col], inplace=True)
    work.sort_values(date_col, kind="mergesort", inplace=True)
    work.reset_index(drop=True, inplace=True)

    if work.empty:
        return _empty_result(initial_capital, allow_short=allow_short)

    closes = work[close_col].to_numpy(dtype=float)
    dates = work[date_col]
    if allow_short:
        signals = [_normalize_signal_intraday(x) for x in work[signal_col].tolist()]
    else:
        signals = [_normalize_signal(x) for x in work[signal_col].tolist()]

    if stop_loss_pct is not None and stop_loss_pct < 0:
        raise ValueError("stop_loss_pct must be non-negative or None.")

    if allow_short:
        trades = _simulate_trades_long_short(
            closes=closes,
            dates=dates,
            signals=signals,
            stop_loss_pct=stop_loss_pct,
        )
        equity_df = _build_equity_curve_long_short(
            dates=dates,
            closes=closes,
            signals=signals,
            initial_capital=float(initial_capital),
            stop_loss_pct=stop_loss_pct,
        )
    else:
        trades = _simulate_trades(
            closes=closes,
            dates=dates,
            signals=signals,
            stop_loss_pct=stop_loss_pct,
        )
        equity_df = _build_equity_curve(
            dates=dates,
            closes=closes,
            signals=signals,
            initial_capital=float(initial_capital),
            stop_loss_pct=stop_loss_pct,
        )

    first_close = float(closes[0])
    last_close = float(closes[-1])
    buy_hold_return = (last_close / first_close - 1.0) * 100.0 if first_close > 0 else 0.0

    final_equity = float(equity_df["equity"].iloc[-1]) if len(equity_df) else float(initial_capital)
    total_return_pct = (final_equity / float(initial_capital) - 1.0) * 100.0 if initial_capital > 0 else 0.0
    strategy_return = total_return_pct

    span_days = max((pd.Timestamp(dates.iloc[-1]) - pd.Timestamp(dates.iloc[0])).days, 1)
    annualized_return_pct = _annualized_return_pct(final_equity, float(initial_capital), span_days)

    winning = [t for t in trades if t["pnl_pct"] > 0]
    losing = [t for t in trades if t["pnl_pct"] <= 0]

    total_trades = len(trades)
    winning_trades = winning
    losing_trades = losing

    win_rate = (len(winning_trades) / total_trades * 100.0) if total_trades else 0.0

    win_pnls = [float(t["pnl_pct"]) for t in winning_trades]
    loss_pnls = [float(t["pnl_pct"]) for t in losing_trades]

    total_wins = float(sum(win_pnls)) if win_pnls else 0.0
    total_losses = float(sum(loss_pnls))  # negative or zero

    avg_win_pct = float(np.mean(win_pnls)) if win_pnls else 0.0
    avg_loss_pct = float(np.mean(loss_pnls)) if loss_pnls else 0.0

    max_win = float(max(win_pnls)) if win_pnls else 0.0
    max_loss = float(min(loss_pnls)) if loss_pnls else 0.0

    gross_profit = sum(win_pnls) if win_pnls else 0.0
    gross_loss_abs = sum(abs(p) for p in loss_pnls) if loss_pnls else 0.0
    if gross_loss_abs == 0:
        profit_factor = math.inf if gross_profit > 0 else float("nan")
    else:
        profit_factor = gross_profit / gross_loss_abs

    loss_rate_pct = 100.0 - win_rate
    avg_loss_mag = abs(avg_loss_pct) if loss_pnls else 0.0
    expected_value = (win_rate / 100.0) * avg_win_pct - (loss_rate_pct / 100.0) * avg_loss_mag

    mfes = [float(t["mfe_pct"]) for t in trades]
    maes = [float(t["mae_pct"]) for t in trades]
    avg_mfe = float(np.mean(mfes)) if mfes else 0.0
    avg_mae = float(np.mean(maes)) if maes else 0.0
    if math.isclose(abs(avg_mae), 0.0, rel_tol=0.0, abs_tol=1e-12):
        mfe_mae_ratio = math.inf if avg_mfe > 0 else float("nan")
    else:
        mfe_mae_ratio = avg_mfe / abs(avg_mae)

    sharpe_ratio, sortino_ratio = _sharpe_sortino(equity_df, dates)

    eq = equity_df["equity"].astype(float)
    max_drawdown = _max_drawdown_pct(eq)

    outperformance = total_return_pct - buy_hold_return

    hold_days = [int(t["holding_days"]) for t in trades]
    avg_holding_days = float(np.mean(hold_days)) if hold_days else 0.0
    max_holding_days = int(max(hold_days)) if hold_days else 0
    min_holding_days = int(min(hold_days)) if hold_days else 0

    max_consecutive_wins, max_consecutive_losses = _consecutive_streaks(trades)

    trades_detail = (
        pd.DataFrame(trades).drop(columns=["entry_idx", "exit_idx"])
        if trades
        else pd.DataFrame(
            columns=[
                "side",
                "entry_date",
                "exit_date",
                "entry_price",
                "exit_price",
                "pnl_pct",
                "pnl_abs",
                "holding_days",
                "exit_reason",
                "mfe_pct",
                "mae_pct",
            ]
        )
    )

    def _pf_out() -> float | None:
        if profit_factor is math.inf or (isinstance(profit_factor, float) and math.isinf(profit_factor)):
            return None
        if isinstance(profit_factor, float) and math.isnan(profit_factor):
            return None
        return float(profit_factor)

    def _mmr_out() -> float | None:
        if mfe_mae_ratio is math.inf or (isinstance(mfe_mae_ratio, float) and math.isinf(mfe_mae_ratio)):
            return None
        if isinstance(mfe_mae_ratio, float) and math.isnan(mfe_mae_ratio):
            return None
        return float(mfe_mae_ratio)

    out: dict[str, Any] = {
        # ── Overall Performance ────────────────────────────────────────────────
        # Total percentage gain/loss on initial capital over the entire period.
        # e.g. 25.0 means the strategy turned ₹1,00,000 into ₹1,25,000.
        "total_return_pct": float(total_return_pct),
        # ── Trade Statistics ───────────────────────────────────────────────────
        # Total number of completed trades (each buy→sell pair counts as 1 trade).
        "total_trades": len(trades),
        # Number of trades that closed with a profit (exit_price > entry_price).
        "winning_trades": len(winning_trades),
        # Number of trades that closed with a loss or broke even (exit_price <= entry_price).
        "losing_trades": len(losing_trades),
        # Percentage of trades that were profitable.
        # e.g. 60.0 means 6 out of every 10 trades made money.
        # A win rate above 50% does NOT guarantee profitability — it must be read
        # alongside avg_win_pct and avg_loss_pct.
        "win_rate": float(win_rate),
        # ── Profit / Loss Analysis ─────────────────────────────────────────────
        # Sum of pnl_pct across all winning trades, expressed as % of initial capital.
        # Represents the total gross profit the strategy generated.
        "total_wins_pct": float(total_wins),
        # Sum of pnl_pct across all losing trades (always positive here for readability).
        # Represents the total gross loss the strategy suffered.
        "total_losses_pct": float(abs(total_losses)),
        # Average profit on a winning trade as % of initial capital.
        # e.g. 4.5 means winning trades made ~4.5% on average.
        "avg_win_pct": float(avg_win_pct),
        # Average loss on a losing trade as % of initial capital (shown as positive).
        # e.g. 2.0 means losing trades lost ~2.0% on average.
        # Ideally avg_win_pct should be significantly larger than avg_loss_pct.
        "avg_loss_pct": float(abs(avg_loss_pct)),
        # Single best trade's profit as % of initial capital.
        "max_win_pct": float(max_win),
        # Single worst trade's loss as % of initial capital (shown as positive).
        "max_loss_pct": float(abs(max_loss)),
        # Gross profit divided by gross loss.
        # e.g. 2.0 means the strategy made ₹2 for every ₹1 it lost.
        # < 1.0 → losing strategy overall. > 1.5 is generally considered good.
        # None if there were no losing trades (division by zero avoided).
        "profit_factor": _pf_out(),
        # Statistical edge per trade = (win_rate% × avg_win) + (loss_rate% × avg_loss).
        # Positive value → strategy has a mathematical edge over time.
        # e.g. 1.2 means on average each trade is expected to return 1.2% of capital.
        "expected_value_pct": float(expected_value),
        # ── Trade Management ───────────────────────────────────────────────────
        # Average Maximum Favourable Excursion — how far price moved IN your favour
        # (from entry) before the trade closed, averaged across all trades.
        # A large avg_mfe vs avg_win gap means you are exiting too early and
        # leaving profits on the table.
        "avg_mfe_pct": float(avg_mfe),
        # Average Maximum Adverse Excursion — how far price moved AGAINST you
        # (from entry) before the trade closed, averaged across all trades.
        # Useful for calibrating stop-loss levels. If avg_mae is -1.5% but your
        # stop is at -3%, you may be giving trades too much room.
        "avg_mae_pct": float(avg_mae),
        # Ratio of avg_mfe to abs(avg_mae).
        # e.g. 2.0 means trades typically moved twice as far in your favour as against.
        # Higher is better. < 1.0 means adverse moves dominate — review your entries.
        # None if avg_mae is zero (no adverse movement at all).
        "mfe_mae_ratio": _mmr_out(),
        # ── Risk Metrics ───────────────────────────────────────────────────────
        # Annualised return divided by annualised volatility of daily equity returns.
        # Measures return per unit of total risk. Higher is better.
        # < 0 → losing strategy. 0–1 → acceptable. > 1 → good. > 2 → excellent.
        "sharpe_ratio": float(sharpe_ratio),
        # Like Sharpe but only penalises DOWNSIDE volatility (losses), not upside.
        # A fairer measure when return distribution is skewed.
        # Generally higher than Sharpe. > 1.5 is considered strong.
        "sortino_ratio": float(sortino_ratio),
        # Largest peak-to-trough decline in the equity curve, as a percentage.
        # e.g. 15.0 means at its worst point the portfolio fell 15% from its peak.
        # Critical for position sizing — you must be able to stomach this drawdown.
        "max_drawdown_pct": float(max_drawdown),
        # Total return scaled to a 1-year period using compounding.
        # Allows fair comparison across strategies tested on different time windows.
        # e.g. a 10% total return over 6 months → ~21% annualised.
        "annualized_return_pct": float(annualized_return_pct),
        # ── Benchmark Comparison ───────────────────────────────────────────────
        # What a passive investor would have earned by buying at the first bar
        # and holding until the last bar — no trading at all.
        "buy_hold_return_pct": float(buy_hold_return),
        # The strategy's total return (same as total_return_pct, included here
        # for side-by-side comparison with buy_hold_return_pct).
        "strategy_return_pct": float(strategy_return),
        # strategy_return_pct minus buy_hold_return_pct.
        # Positive → your strategy beat passive holding.
        # Negative → you would have done better just buying and holding.
        "outperformance_pct": float(outperformance),
        # ── Trading Behaviour ──────────────────────────────────────────────────
        # Average number of calendar days a position was held open.
        # Helps classify the strategy: < 5 days = short-term, > 20 days = swing/positional.
        "avg_holding_days": float(avg_holding_days),
        # Longest a single trade was held open (calendar days).
        "max_holding_days": int(max_holding_days),
        # Shortest a single trade was held open (calendar days).
        "min_holding_days": int(min_holding_days),
        # Longest unbroken streak of back-to-back profitable trades.
        # Useful for understanding if wins tend to cluster together.
        "max_consecutive_wins": int(max_consecutive_wins),
        # Longest unbroken streak of back-to-back losing trades.
        # Key for psychological/risk tolerance assessment — can you survive
        # N losses in a row without abandoning the strategy?
        "max_consecutive_losses": int(max_consecutive_losses),
        # ── Detailed Trade Log ─────────────────────────────────────────────────
        # DataFrame where each row is one completed trade. Columns:
        #
        #   entry_date   — Date the buy signal triggered and position was opened.
        #   exit_date    — Date the position was closed (sell signal or stop loss hit).
        #   entry_price  — Close price at which the trade was entered.
        #   exit_price   — Close price at which the trade was exited.
        #   pnl_pct      — Profit or loss as % of entry price.
        #                  Positive = profit, Negative = loss.
        #                  e.g. 3.5 means the trade made 3.5% from entry to exit.
        #   pnl_abs      — Raw price difference (exit_price - entry_price).
        #                  Useful when position sizing in absolute terms.
        #   holding_days — Calendar days the trade was open (exit_date - entry_date).
        #   exit_reason  — Why the trade was closed:
        #                    'signal'    → a SELL signal appeared in the DataFrame.
        #                    'stop_loss' → price fell below entry × (1 - stop_loss_pct%).
        #                    'open'      → trade was still open at the last bar;
        #                                  closed at last available price.
        #   mfe_pct      — Maximum Favourable Excursion: the highest % gain the trade
        #                  reached at any point before closing.
        #                  e.g. mfe_pct=5.0 but pnl_pct=2.0 means price went up 5%
        #                  but you exited at only +2% — possible early exit.
        #   mae_pct      — Maximum Adverse Excursion: the deepest % loss the trade
        #                  experienced at any point before closing (always <= 0).
        #                  e.g. mae_pct=-1.5 means price dipped 1.5% against you
        #                  at its worst, even if the trade ultimately closed profitably.
        "trades_detail": trades_detail,
        # ── Equity Curve ───────────────────────────────────────────────────────
        # DataFrame with columns ['date', 'equity'].
        # Shows how the portfolio value evolved bar-by-bar starting from
        # initial_capital: cash when flat; when long, mark-to-market as
        # entry_equity × (close / entry_price). Compounds across trades when
        # re-entering. Use this to plot the growth curve and inspect drawdowns.
        "equity_curve": equity_df,
    }
    if allow_short:
        out.update(_intraday_side_breakdown(trades))
    return out


def _intraday_side_breakdown(trades: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-side trade stats (only merged when ``allow_short=True``)."""
    longs = [t for t in trades if t.get("side") == "long"]
    shorts = [t for t in trades if t.get("side") == "short"]
    z = 0.0

    def avg_pnl(ts: list[dict[str, Any]]) -> float:
        if not ts:
            return z
        return float(np.mean([float(t["pnl_pct"]) for t in ts]))

    def win_rate(ts: list[dict[str, Any]]) -> float:
        if not ts:
            return z
        w = sum(1 for t in ts if float(t["pnl_pct"]) > 0)
        return 100.0 * w / len(ts)

    def profit_factor(ts: list[dict[str, Any]]) -> float | None:
        wins = [float(t["pnl_pct"]) for t in ts if float(t["pnl_pct"]) > 0]
        losses = [float(t["pnl_pct"]) for t in ts if float(t["pnl_pct"]) <= 0]
        gp = float(sum(wins)) if wins else 0.0
        gl = float(sum(abs(p) for p in losses)) if losses else 0.0
        if gl == 0.0:
            return math.inf if gp > 0.0 else None
        return gp / gl

    l_pf = profit_factor(longs)
    s_pf = profit_factor(shorts)

    return {
        "long_trades": int(len(longs)),
        "short_trades": int(len(shorts)),
        "long_win_rate": float(win_rate(longs)),
        "short_win_rate": float(win_rate(shorts)),
        "long_avg_trade_pnl_pct": float(avg_pnl(longs)),
        "short_avg_trade_pnl_pct": float(avg_pnl(shorts)),
        "long_profit_factor": None if l_pf is None or math.isinf(l_pf) or math.isnan(l_pf) else float(l_pf),
        "short_profit_factor": None if s_pf is None or math.isinf(s_pf) or math.isnan(s_pf) else float(s_pf),
        "long_total_pnl_pct": float(sum(float(t["pnl_pct"]) for t in longs)),
        "short_total_pnl_pct": float(sum(float(t["pnl_pct"]) for t in shorts)),
    }


def _empty_result(initial_capital: float, *, allow_short: bool = False) -> dict[str, Any]:
    eq = pd.DataFrame(columns=["date", "equity"])
    empty_trades = pd.DataFrame(
        columns=[
            "side",
            "entry_date",
            "exit_date",
            "entry_price",
            "exit_price",
            "pnl_pct",
            "pnl_abs",
            "holding_days",
            "exit_reason",
            "mfe_pct",
            "mae_pct",
        ]
    )
    z = 0.0
    out: dict[str, Any] = {
        "total_return_pct": z,
        "total_trades": 0,
        "winning_trades": 0,
        "losing_trades": 0,
        "win_rate": z,
        "total_wins_pct": z,
        "total_losses_pct": z,
        "avg_win_pct": z,
        "avg_loss_pct": z,
        "max_win_pct": z,
        "max_loss_pct": z,
        "profit_factor": None,
        "expected_value_pct": z,
        "avg_mfe_pct": z,
        "avg_mae_pct": z,
        "mfe_mae_ratio": None,
        "sharpe_ratio": z,
        "sortino_ratio": z,
        "max_drawdown_pct": z,
        "annualized_return_pct": z,
        "buy_hold_return_pct": z,
        "strategy_return_pct": z,
        "outperformance_pct": z,
        "avg_holding_days": z,
        "max_holding_days": 0,
        "min_holding_days": 0,
        "max_consecutive_wins": 0,
        "max_consecutive_losses": 0,
        "trades_detail": empty_trades,
        "equity_curve": eq,
    }
    if allow_short:
        out.update(_intraday_side_breakdown([]))
    return out


def _simulate_trades(
    *,
    closes: np.ndarray,
    dates: pd.Series,
    signals: list[str],
    stop_loss_pct: float | None,
) -> list[dict[str, Any]]:
    """Long-only: BUY opens, SELL / stop / last bar closes. Stop is checked before SELL on the same bar."""
    trades: list[dict[str, Any]] = []
    in_pos = False
    entry_price = 0.0
    entry_date: pd.Timestamp | None = None
    entry_idx = -1
    n = len(closes)

    for i in range(n):
        close = float(closes[i])
        dt = pd.Timestamp(dates.iloc[i])
        sig = signals[i]

        if not in_pos:
            if sig == "buy":
                in_pos = True
                entry_price = close
                entry_date = dt
                entry_idx = i
            continue

        stop_hit = False
        if stop_loss_pct is not None:
            thr = entry_price * (1.0 - float(stop_loss_pct) / 100.0)
            if close <= thr:
                stop_hit = True

        if stop_hit:
            trades.append(
                _finalize_trade(
                    entry_idx=entry_idx,
                    exit_idx=i,
                    entry_date=entry_date,
                    exit_date=dt,
                    entry_price=entry_price,
                    exit_price=close,
                    exit_reason="stop_loss",
                    closes=closes,
                    side="long",
                )
            )
            in_pos = False
            continue

        if sig == "sell":
            trades.append(
                _finalize_trade(
                    entry_idx=entry_idx,
                    exit_idx=i,
                    entry_date=entry_date,
                    exit_date=dt,
                    entry_price=entry_price,
                    exit_price=close,
                    exit_reason="signal",
                    closes=closes,
                    side="long",
                )
            )
            in_pos = False
            continue

    if in_pos and entry_date is not None and entry_idx >= 0:
        last_i = n - 1
        trades.append(
            _finalize_trade(
                entry_idx=entry_idx,
                exit_idx=last_i,
                entry_date=entry_date,
                exit_date=pd.Timestamp(dates.iloc[last_i]),
                entry_price=entry_price,
                exit_price=float(closes[last_i]),
                exit_reason="open",
                closes=closes,
                side="long",
            )
        )

    return trades


def _finalize_trade(
    *,
    entry_idx: int,
    exit_idx: int,
    entry_date: pd.Timestamp | None,
    exit_date: pd.Timestamp,
    entry_price: float,
    exit_price: float,
    exit_reason: str,
    closes: np.ndarray,
    side: str = "long",
) -> dict[str, Any]:
    window = closes[entry_idx : exit_idx + 1]
    if side == "long":
        rets = (window / entry_price - 1.0) * 100.0
        mfe_pct = float(np.max(rets)) if len(rets) else 0.0
        mae_pct = float(np.min(rets)) if len(rets) else 0.0
        pnl_abs = float(exit_price - entry_price)
        pnl_pct = float((exit_price / entry_price - 1.0) * 100.0) if entry_price > 0 else 0.0
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            rets = (entry_price / window - 1.0) * 100.0
        rets = np.nan_to_num(rets, nan=0.0, posinf=0.0, neginf=0.0)
        mfe_pct = float(np.max(rets)) if len(rets) else 0.0
        mae_pct = float(np.min(rets)) if len(rets) else 0.0
        pnl_abs = float(entry_price - exit_price)
        pnl_pct = float((entry_price - exit_price) / entry_price * 100.0) if entry_price > 0 else 0.0
    ed = pd.Timestamp(entry_date) if entry_date is not None else pd.Timestamp(exit_date)
    holding_days = int(max((exit_date.normalize() - ed.normalize()).days, 0))
    return {
        "entry_idx": entry_idx,
        "exit_idx": exit_idx,
        "side": side,
        "entry_date": ed,
        "exit_date": exit_date,
        "entry_price": float(entry_price),
        "exit_price": float(exit_price),
        "pnl_pct": pnl_pct,
        "pnl_abs": pnl_abs,
        "holding_days": holding_days,
        "exit_reason": exit_reason,
        "mfe_pct": mfe_pct,
        "mae_pct": mae_pct,
    }


def _build_equity_curve(
    dates: pd.Series,
    closes: np.ndarray,
    signals: list[str],
    initial_capital: float,
    stop_loss_pct: float | None,
) -> pd.DataFrame:
    """Full-length equity: cash when flat; MTM ``entry_equity * close/entry`` when long."""
    in_pos = False
    entry_price = 0.0
    entry_equity = 0.0
    curve: list[float] = []

    for i in range(len(closes)):
        close = float(closes[i])
        sig = signals[i]

        if not in_pos:
            eq = initial_capital if not curve else float(curve[-1])
            if sig == "buy":
                in_pos = True
                entry_price = close
                entry_equity = eq
            curve.append(eq)
            continue

        eq = float(entry_equity * (close / entry_price))
        stop_hit = False
        if stop_loss_pct is not None:
            thr = entry_price * (1.0 - float(stop_loss_pct) / 100.0)
            if close <= thr:
                stop_hit = True
        if stop_hit or sig == "sell":
            in_pos = False
        curve.append(eq)

    return pd.DataFrame({"date": pd.to_datetime(dates, errors="coerce").values, "equity": curve})


def _simulate_trades_long_short(
    *,
    closes: np.ndarray,
    dates: pd.Series,
    signals: list[str],
    stop_loss_pct: float | None,
) -> list[dict[str, Any]]:
    """``long_buy``/``long_sell`` and ``short_sell``/``short_buy``; one position at a time."""
    trades: list[dict[str, Any]] = []
    state: str = "flat"
    entry_price = 0.0
    entry_date: pd.Timestamp | None = None
    entry_idx = -1
    n = len(closes)

    for i in range(n):
        close = float(closes[i])
        dt = pd.Timestamp(dates.iloc[i])
        sig = signals[i]

        if state == "flat":
            if sig == "long_buy":
                state = "long"
                entry_price = close
                entry_date = dt
                entry_idx = i
            elif sig == "short_sell":
                state = "short"
                entry_price = close
                entry_date = dt
                entry_idx = i
            continue

        if state == "long":
            stop_hit = False
            if stop_loss_pct is not None:
                thr = entry_price * (1.0 - float(stop_loss_pct) / 100.0)
                if close <= thr:
                    stop_hit = True
            if stop_hit:
                trades.append(
                    _finalize_trade(
                        entry_idx=entry_idx,
                        exit_idx=i,
                        entry_date=entry_date,
                        exit_date=dt,
                        entry_price=entry_price,
                        exit_price=close,
                        exit_reason="stop_loss",
                        closes=closes,
                        side="long",
                    )
                )
                state = "flat"
                continue
            if sig == "long_sell":
                trades.append(
                    _finalize_trade(
                        entry_idx=entry_idx,
                        exit_idx=i,
                        entry_date=entry_date,
                        exit_date=dt,
                        entry_price=entry_price,
                        exit_price=close,
                        exit_reason="signal",
                        closes=closes,
                        side="long",
                    )
                )
                state = "flat"
                continue
            continue

        if state == "short":
            stop_hit = False
            if stop_loss_pct is not None:
                thr = entry_price * (1.0 + float(stop_loss_pct) / 100.0)
                if close >= thr:
                    stop_hit = True
            if stop_hit:
                trades.append(
                    _finalize_trade(
                        entry_idx=entry_idx,
                        exit_idx=i,
                        entry_date=entry_date,
                        exit_date=dt,
                        entry_price=entry_price,
                        exit_price=close,
                        exit_reason="stop_loss",
                        closes=closes,
                        side="short",
                    )
                )
                state = "flat"
                continue
            if sig == "short_buy":
                trades.append(
                    _finalize_trade(
                        entry_idx=entry_idx,
                        exit_idx=i,
                        entry_date=entry_date,
                        exit_date=dt,
                        entry_price=entry_price,
                        exit_price=close,
                        exit_reason="signal",
                        closes=closes,
                        side="short",
                    )
                )
                state = "flat"
                continue
            continue

    if state == "long" and entry_date is not None and entry_idx >= 0:
        last_i = n - 1
        trades.append(
            _finalize_trade(
                entry_idx=entry_idx,
                exit_idx=last_i,
                entry_date=entry_date,
                exit_date=pd.Timestamp(dates.iloc[last_i]),
                entry_price=entry_price,
                exit_price=float(closes[last_i]),
                exit_reason="open",
                closes=closes,
                side="long",
            )
        )
    elif state == "short" and entry_date is not None and entry_idx >= 0:
        last_i = n - 1
        trades.append(
            _finalize_trade(
                entry_idx=entry_idx,
                exit_idx=last_i,
                entry_date=entry_date,
                exit_date=pd.Timestamp(dates.iloc[last_i]),
                entry_price=entry_price,
                exit_price=float(closes[last_i]),
                exit_reason="open",
                closes=closes,
                side="short",
            )
        )

    return trades


def _build_equity_curve_long_short(
    dates: pd.Series,
    closes: np.ndarray,
    signals: list[str],
    initial_capital: float,
    stop_loss_pct: float | None,
) -> pd.DataFrame:
    """Cash when flat; long MTM ``entry_equity * close/entry``; short ``entry_equity * entry/close``."""
    state: str = "flat"
    entry_price = 0.0
    entry_equity = 0.0
    curve: list[float] = []

    for i in range(len(closes)):
        close = float(closes[i])
        sig = signals[i]

        if state == "flat":
            eq = initial_capital if not curve else float(curve[-1])
            if sig == "long_buy":
                state = "long"
                entry_price = close
                entry_equity = eq
            elif sig == "short_sell":
                state = "short"
                entry_price = close
                entry_equity = eq
            curve.append(eq)
            continue

        if state == "long":
            eq = float(entry_equity * (close / entry_price))
            stop_hit = False
            if stop_loss_pct is not None:
                thr = entry_price * (1.0 - float(stop_loss_pct) / 100.0)
                if close <= thr:
                    stop_hit = True
            if stop_hit or sig == "long_sell":
                state = "flat"
            curve.append(eq)
            continue

        if state == "short":
            eq = float(entry_equity * (entry_price / close)) if close > 0 else float(curve[-1])
            stop_hit = False
            if stop_loss_pct is not None:
                thr = entry_price * (1.0 + float(stop_loss_pct) / 100.0)
                if close >= thr:
                    stop_hit = True
            if stop_hit or sig == "short_buy":
                state = "flat"
            curve.append(eq)
            continue

    return pd.DataFrame({"date": pd.to_datetime(dates, errors="coerce").values, "equity": curve})


def _sharpe_sortino(equity_df: pd.DataFrame, dates: pd.Series) -> tuple[float, float]:
    if equity_df is None or len(equity_df) < 3:
        return 0.0, 0.0
    r = equity_df["equity"].pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    if r.empty or float(r.std()) == 0.0:
        return 0.0, 0.0
    ppy = _periods_per_year(pd.to_datetime(dates, errors="coerce"))
    mu = float(r.mean())
    sd = float(r.std())
    sharpe = (mu / sd) * math.sqrt(ppy) if sd > 0 else 0.0

    neg = r[r < 0]
    if neg.empty:
        sortino = sharpe
    else:
        dsd = float(neg.std())
        sortino = (mu / dsd) * math.sqrt(ppy) if dsd > 0 else 0.0
    return float(sharpe), float(sortino)


def _max_drawdown_pct(eq: pd.Series) -> float:
    if eq.empty:
        return 0.0
    cummax = eq.cummax()
    dd = (eq / cummax - 1.0) * 100.0
    return float(abs(dd.min())) if len(dd) else 0.0


def _consecutive_streaks(trades: list[dict[str, Any]]) -> tuple[int, int]:
    max_w = max_l = cur_w = cur_l = 0
    for t in trades:
        win = float(t["pnl_pct"]) > 0
        if win:
            cur_w += 1
            cur_l = 0
            max_w = max(max_w, cur_w)
        else:
            cur_l += 1
            cur_w = 0
            max_l = max(max_l, cur_l)
    return max_w, max_l
