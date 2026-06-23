from __future__ import annotations

import asyncio
import functools
import html
import logging
import os
from pathlib import Path

from telegram import InputFile, InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.error import BadRequest
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from trade_bot.config import Config, load_config
from trade_bot.services.allowlist_store import read_extra_user_ids, write_extra_user_ids
from trade_bot.services.backtest_all import backtest_all_csv_path, run_backtest_all_module_async
from trade_bot.services.backtest_single import (
    SingleStockAnalysis,
    analyze_single_stock,
    analyze_single_stock_from_saved_params,
)
from trade_bot.services.enrich import merge_bot_stock_metadata, optional_company_names
from trade_bot.services.global_jobs import GlobalJobCoordinator, clear_stale_job_busy_files
from trade_bot.services.inputs import bot_stocks_path, ensure_input_stocks, load_symbol_universe
from trade_bot.services.live_signals import (
    enrich_live_signals_csv_with_fib,
    live_signals_csv_path,
    materialize_live_signals_upload_path,
    run_live_signals_module_async,
)
from trade_bot.services.stock_filters import (
    distinct_column_sorted,
    filters_for_subprocess,
    parse_filters_from_command_args,
)
from trade_bot.services.subprocess_stream import format_stream_blob
from trade_bot.services.telegram_files import prepare_csv_for_telegram
from trade_bot.services.research_tools import (
    compute_fib_package,
    fetch_screener_bundle,
    fib_package_to_text,
)

logging.basicConfig(
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    level=logging.INFO,
)
# Bot uses httpx for every Telegram API call; INFO prints each POST (getUpdates, editMessageText, …).
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
log = logging.getLogger("trade_bot")


def _log_command_lifecycle(command: str, handler):
    """Log when a slash-command starts / finishes (with full command text)."""

    async def wrapped(update: Update, context: ContextTypes.DEFAULT_TYPE):
        uid = update.effective_user.id if update.effective_user else None
        m = update.effective_message
        txt = (m.text or "").strip()[:400] if m else ""
        log.info("/%s start user_id=%s text=%r", command, uid, txt)
        try:
            return await handler(update, context)
        finally:
            log.info("/%s done user_id=%s", command, uid)

    return wrapped


async def _log_non_command_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Log ordinary chat text (not slash-commands) at INFO for the host console."""
    msg = update.effective_message
    if not msg or not msg.text:
        return
    if msg.text.strip().startswith("/"):
        return
    uid = update.effective_user.id if update.effective_user else None
    log.info("incoming_text chat_id=%s user_id=%s %r", msg.chat_id, uid, msg.text[:500])


def _clamp_telegram_html(html: str, max_chars: int = 4096) -> str:
    """Keep ``<pre>…</pre>`` well-formed when trimming to Telegram's length limit."""
    if len(html) <= max_chars:
        return html
    close = html.rfind("</pre>")
    open_ = html.rfind("<pre>", 0, close if close != -1 else len(html))
    if open_ == -1 or close == -1 or close < open_:
        return html[: max_chars - 1] + "…"
    prefix = html[: open_ + 5]
    suffix = html[close:]
    inner = html[open_ + 5 : close]
    budget = max_chars - len(prefix) - len(suffix) - 2
    budget = max(budget, 40)
    if len(inner) <= budget:
        return prefix + inner + suffix
    trimmed = "…\n" + inner[-(budget - 2) :]
    return prefix + trimmed + suffix


def _split_plain_text_for_pre(text: str, max_inner: int = 3500) -> list[str]:
    """Split long plain text so each chunk fits under Telegram limits with a <pre> wrapper."""
    text = text.strip()
    if len(text) <= max_inner:
        return [text]
    chunks: list[str] = []
    rest = text
    while rest:
        if len(rest) <= max_inner:
            chunks.append(rest)
            break
        cut = rest.rfind("\n", 0, max_inner)
        if cut < max_inner // 3:
            cut = max_inner
        chunks.append(rest[:cut].rstrip())
        rest = rest[cut:].lstrip("\n")
    return chunks


def _analysis_summary_html(res: SingleStockAnalysis) -> str:
    """Symbol, params, current signal, then one <pre> with STOCKS.csv fields + all scalar metrics."""
    sig = (res.latest_signal or "hold").upper()
    header = [
        f"<b>{html.escape(res.symbol)}</b> — {html.escape(res.strategy_label)}",
        f"best_short_ma={res.best_short_ma}  best_long_ma={res.best_long_ma}  optimization_score={res.optimization_score}",
        "",
        f"<b>Current signal (latest bar): {html.escape(sig)}</b>",
    ]
    if res.signal_bar_date:
        header.append(f"Latest bar time: {html.escape(res.signal_bar_date)}")
    if res.signal_last_close is not None:
        header.append(f"Latest bar close: {res.signal_last_close}")
    body = (
        "=== Stock info (input/STOCKS.csv) ===\n"
        + (res.stock_metadata_block or "")
        + "\n\n=== Backtest metrics (all scalars) ===\n"
        + (res.metrics_block or "")
    )
    return "\n".join(header) + f"\n\n<pre>{html.escape(body)}</pre>"


async def _send_analysis_bundle(msg, res: SingleStockAnalysis, sym: str) -> None:
    """Summary + trades preview (readable) + optional equity chart + full trades CSV."""
    await msg.reply_text(_clamp_telegram_html(_analysis_summary_html(res), 4096), parse_mode="HTML")
    trades_pre = (
        "<b>Completed trades (preview, one block per trade)</b>\n"
        f"<pre>{html.escape(res.last_trades_table)}</pre>"
    )
    await msg.reply_text(_clamp_telegram_html(trades_pre, 4096), parse_mode="HTML")
    if res.equity_png:
        await msg.reply_photo(
            photo=InputFile(res.equity_png, filename="equity.png"),
            caption=f"{sym} equity curve ({res.completed_trade_count} completed trades)",
        )
    if res.trades_csv:
        await msg.reply_document(
            document=InputFile(res.trades_csv, filename=f"{sym}_all_trades.csv"),
            caption=f"Full trade log: {res.completed_trade_count} rows (open in Excel/Sheets)",
        )


WORK_SUBDIR = Path("_telegram_work")


async def _telegram_edit_stream_message(bot, chat_id: int, message_id: int, html_text: str) -> None:
    """Edit one status message; tolerate 'not modified', length limits, and transient network errors."""
    html_text = _clamp_telegram_html(html_text, 4096)
    try:
        await bot.edit_message_text(
            chat_id=chat_id, message_id=message_id, text=html_text, parse_mode="HTML"
        )
    except BadRequest as exc:
        low = (exc.message or str(exc)).lower()
        if "not modified" in low or "message is not modified" in low:
            return
        try:
            await bot.edit_message_text(
                chat_id=chat_id, message_id=message_id, text=html_text[:4096], parse_mode=None
            )
        except BadRequest:
            pass
    except Exception as exc:
        # Long jobs (live_signals / backtest_all) can hit connect timeouts; must not kill the worker.
        log.warning("edit_message_text failed (progress update skipped): %s", exc)
        try:
            await bot.edit_message_text(
                chat_id=chat_id, message_id=message_id, text=html_text[:4096], parse_mode=None
            )
        except Exception:
            pass


def _cfg(context: ContextTypes.DEFAULT_TYPE) -> Config:
    return context.application.bot_data["config"]


def _job_coordinator(context: ContextTypes.DEFAULT_TYPE) -> GlobalJobCoordinator:
    return context.application.bot_data["jobs"]


def _strategy(context: ContextTypes.DEFAULT_TYPE) -> str | None:
    s = context.user_data.get("strategy")
    if s in ("sma", "ema"):
        return s
    return None


async def _ensure_strategy(update: Update, context: ContextTypes.DEFAULT_TYPE) -> str | None:
    s = _strategy(context)
    if s:
        return s
    msg = update.effective_message
    if msg:
        await msg.reply_text("Choose a strategy first: send /start and tap **SMA** or **EMA**.")
    return None


async def _ensure_allowed(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    cfg = _cfg(context)
    uid = update.effective_user.id if update.effective_user else None
    if not cfg.is_user_allowed(uid):
        if update.effective_message:
            await update.effective_message.reply_text("You are not allowed to use this bot.")
        return False
    return True


def _is_admin(cfg: Config, user_id: int | None) -> bool:
    if user_id is None:
        return False
    return int(user_id) == int(cfg.admin_user_id)


async def _admin_unknown_reply(msg) -> None:
    """Do not hint that privileged commands exist."""
    await msg.reply_text("Unknown command.")


async def cmd_admin_add(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = update.effective_message
    if not msg:
        return
    cfg = _cfg(context)
    uid = update.effective_user.id if update.effective_user else None
    if not _is_admin(cfg, uid):
        await _admin_unknown_reply(msg)
        return
    if not context.args:
        await msg.reply_text("Usage: `/admin_add <user_id>`", parse_mode="Markdown")
        return
    try:
        new_id = int(context.args[0].strip())
    except ValueError:
        await msg.reply_text("Invalid user id.")
        return
    if new_id <= 0:
        await msg.reply_text("Invalid user id.")
        return
    lock = context.application.bot_data["allowlist_lock"]
    async with lock:
        cur = read_extra_user_ids(cfg.project_root)
        if new_id in cur:
            await msg.reply_text(f"User <code>{new_id}</code> is already in the bot list.", parse_mode="HTML")
            return
        cur.add(new_id)
        write_extra_user_ids(cfg.project_root, cur)
    log.info("allowlist add new_id=%s by_admin=%s", new_id, uid)
    await msg.reply_text(f"Added user <code>{new_id}</code>.", parse_mode="HTML")


async def cmd_admin_remove(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = update.effective_message
    if not msg:
        return
    cfg = _cfg(context)
    uid = update.effective_user.id if update.effective_user else None
    if not _is_admin(cfg, uid):
        await _admin_unknown_reply(msg)
        return
    if not context.args:
        await msg.reply_text("Usage: `/admin_remove <user_id>`", parse_mode="Markdown")
        return
    try:
        rid = int(context.args[0].strip())
    except ValueError:
        await msg.reply_text("Invalid user id.")
        return
    if rid <= 0:
        await msg.reply_text("Invalid user id.")
        return
    lock = context.application.bot_data["allowlist_lock"]
    async with lock:
        cur = read_extra_user_ids(cfg.project_root)
        if rid not in cur:
            await msg.reply_text(
                "That id is not in the bot-managed list. "
                "(Users listed only in <code>ALLOWED_USER_IDS</code> on the server must be edited in <code>.env</code>.)",
                parse_mode="HTML",
            )
            return
        cur.discard(rid)
        write_extra_user_ids(cfg.project_root, cur)
    log.info("allowlist remove rid=%s by_admin=%s", rid, uid)
    await msg.reply_text(f"Removed user <code>{rid}</code> from the bot list.", parse_mode="HTML")


async def cmd_admin_list(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = update.effective_message
    if not msg:
        return
    cfg = _cfg(context)
    uid = update.effective_user.id if update.effective_user else None
    if not _is_admin(cfg, uid):
        await _admin_unknown_reply(msg)
        return
    env_ids = sorted(cfg.allowed_user_ids)
    disk = sorted(read_extra_user_ids(cfg.project_root))
    open_all = not env_ids and not disk
    lines = [
        "<b>Access control</b>",
        f"Admin id: <code>{cfg.admin_user_id}</code>",
        f"Open to any Telegram user (no env or bot list): <b>{'yes' if open_all else 'no'}</b>",
        "",
        "<b>ALLOWED_USER_IDS</b> (.env): "
        + (html.escape(", ".join(str(x) for x in env_ids)) if env_ids else "<i>(none)</i>"),
        "<b>Bot-managed list</b> (file): "
        + (html.escape(", ".join(str(x) for x in disk)) if disk else "<i>(none)</i>"),
    ]
    await msg.reply_text("\n".join(lines), parse_mode="HTML")


def _strategy_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("SMA crossover (daily)", callback_data="strategy:sma"),
                InlineKeyboardButton("EMA crossover (15m)", callback_data="strategy:ema"),
            ]
        ]
    )


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _ensure_allowed(update, context):
        return
    msg = update.effective_message
    if not msg:
        return
    await msg.reply_text(
        "Hi. Pick which engine to drive (you can change later with /strategy):\n\n"
        "• **SMA** — daily bars, `brute_sma_cross`\n"
        "• **EMA** — 15-minute bars, `brute_ema_cross`\n\n"
        "Your universe is always this repo's `input/STOCKS.csv` (read directly by the bundled engine).",
        parse_mode="Markdown",
        reply_markup=_strategy_keyboard(),
    )


async def cmd_strategy(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _ensure_allowed(update, context):
        return
    msg = update.effective_message
    if not msg:
        return
    cur = _strategy(context) or "not set yet"
    await msg.reply_text(
        f"Current strategy: **{cur}**\nPick one:",
        parse_mode="Markdown",
        reply_markup=_strategy_keyboard(),
    )


async def on_strategy_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _ensure_allowed(update, context):
        return
    q = update.callback_query
    if not q:
        return
    uid = update.effective_user.id if update.effective_user else None
    log.info("strategy_inline user_id=%s data=%r", uid, (q.data or "")[:80])
    await q.answer()
    data = (q.data or "").strip()
    if not data.startswith("strategy:"):
        return
    _, strat = data.split(":", 1)
    if strat not in ("sma", "ema"):
        return
    context.user_data["strategy"] = strat
    label = "SMA / daily (`brute_sma_cross`)" if strat == "sma" else "EMA / 15m (`brute_ema_cross`)"
    await q.edit_message_text(f"Strategy set: **{label}**", parse_mode="Markdown")
    cfg = _cfg(context)
    admin = _is_admin(cfg, uid)
    parts = [
        "Commands:\n",
        "/get_stocks /get_sectors /get_industries — universe helpers\n",
    ]
    if admin:
        parts.append("/backtest_all — bulk backtest (administrator only)\n")
    parts.extend(
        [
            "/live_signals [filters…] — latest-bar screen (optional `signal=buy`/`sell`; "
            "`fib=true`/`fib=false`; admins get Fib columns by default)\n",
            "/bt1 SYMBOL — one-symbol optimization + metrics + equity chart\n",
            "/deep SYMBOL — backtest using /backtest_all params + trades preview + CSV\n",
            "/fib SYMBOL — latest swing Fib levels (1d + 15m + 5m Yahoo bars)\n",
            "/screener SYMBOL — full automated read in chat (may split across messages; no files)\n",
            "/help — full command list\n",
        ]
    )
    if admin:
        parts.append("\nAdministrators: /help also lists /admin_add, /admin_remove, /admin_list.\n")
    await context.bot.send_message(
        chat_id=q.message.chat_id,
        text="".join(parts),
    )


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _ensure_allowed(update, context):
        return
    msg = update.effective_message
    if not msg:
        return
    cfg = _cfg(context)
    uid = update.effective_user.id if update.effective_user else None
    admin = _is_admin(cfg, uid)

    lines: list[str] = [
        "/start — pick SMA or EMA",
        "/strategy — change engine",
        "/get_stocks — download `input/STOCKS.csv`",
        "/get_sectors — list distinct sectors (needs `sector` column)",
        "/get_industries — list distinct industries as a file (needs `industry` column)",
    ]
    if admin:
        lines.append(
            "/backtest_all — bulk backtest → `backtest_all_stocks.csv` "
            "(optional: `sector=fmcg` `industry=textiles` `cap=smallcap`)"
        )
    lines.extend(
        [
            "/live_signals — screen latest bar → `live_signals.csv` "
            "(sector/industry/cap; optional `signal=buy`/`sell`; `fib=true`/`false`; "
            "admins: Fib swing columns on by default unless `fib=false`; others: off unless `fib=true`). "
            "Fib adds per-timeframe direction + swing high/low price and time (1d, 15m, 5m).",
            "/bt1 SYMBOL — optimize one symbol + metrics + equity + trades CSV",
            "/deep SYMBOL — same outputs using saved /backtest_all params (no re-optimize)",
            "/fib SYMBOL — Fibonacci retracements from latest qualifying swing on 1d, 15m, and 5m",
            "/screener SYMBOL — long automated analysis in chat only (no attachments)",
        ]
    )
    if admin:
        lines.extend(
            [
                "",
                "Administrator:",
                "/admin_add <user_id> — add a Telegram user id to the bot-managed allowlist",
                "/admin_remove <user_id> — remove an id from the bot-managed list (not .env users)",
                "/admin_list — show admin id, .env allowlist, and bot-managed ids",
            ]
        )
    lines.extend(
        [
            "",
            "Filters use `marketcapname` in CSV: `largecap`, `midcap`, `smallcap`.",
            "Place your universe at `input/STOCKS.csv` (at least `symbol`; optional metadata columns).",
            "",
            "Optional env: `ENRICH_COMPANY_NAMES=1` to fill `company_name` via Yahoo (slow).",
        ]
    )
    await msg.reply_text("\n".join(lines))


def _filter_summary_html(filters: dict[str, str]) -> str:
    if not filters:
        return ""
    parts: list[str] = []
    if v := filters.get("sector"):
        parts.append(f"sector={html.escape(v)}")
    if v := filters.get("industry"):
        parts.append(f"industry={html.escape(v)}")
    if v := filters.get("cap"):
        parts.append(f"cap={html.escape(v)}")
    if v := filters.get("signal"):
        parts.append(f"signal={html.escape(v)}")
    if v := filters.get("fib"):
        parts.append(f"fib={html.escape(v)}")
    return "<i>Filters: " + " ".join(parts) + "</i>\n"


async def cmd_get_stocks(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _ensure_allowed(update, context):
        return
    msg = update.effective_message
    if not msg:
        return
    cfg = _cfg(context)
    p = ensure_input_stocks(cfg.project_root)
    data = p.read_bytes()
    await msg.reply_document(document=InputFile(data, filename="STOCKS.csv"), caption="input/STOCKS.csv")


async def cmd_get_sectors(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _ensure_allowed(update, context):
        return
    msg = update.effective_message
    if not msg:
        return
    cfg = _cfg(context)
    ensure_input_stocks(cfg.project_root)
    sectors = distinct_column_sorted(cfg.project_root, "sector")
    if not sectors:
        await msg.reply_text("No `sector` column in STOCKS.csv (or file empty).")
        return
    body = "\n".join(sectors)
    if len(body) <= 3500:
        await msg.reply_text(f"<pre>{html.escape(body)}</pre>", parse_mode="HTML")
    else:
        raw = body.encode("utf-8")
        await msg.reply_document(
            document=InputFile(raw, filename="sectors.txt"),
            caption=f"{len(sectors)} distinct sectors",
        )


async def cmd_get_industries(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _ensure_allowed(update, context):
        return
    msg = update.effective_message
    if not msg:
        return
    cfg = _cfg(context)
    ensure_input_stocks(cfg.project_root)
    industries = distinct_column_sorted(cfg.project_root, "industry")
    if not industries:
        await msg.reply_text("No `industry` column in STOCKS.csv (or file empty).")
        return
    body = "\n".join(industries)
    raw = body.encode("utf-8")
    await msg.reply_document(
        document=InputFile(raw, filename="industries.txt"),
        caption=f"{len(industries)} distinct industries (file — list is long for chat)",
    )


def _post_backtest_csv_merge(cfg: Config, csv_path: Path) -> None:
    if not csv_path.is_file():
        return
    merge_bot_stock_metadata(cfg.project_root, csv_path)
    if os.environ.get("ENRICH_COMPANY_NAMES", "").strip().lower() in ("1", "true", "yes"):
        optional_company_names(csv_path)


async def cmd_backtest_all(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _ensure_allowed(update, context):
        return
    strat = await _ensure_strategy(update, context)
    if not strat:
        return
    msg = update.effective_message
    if not msg:
        return
    cfg = _cfg(context)
    uid = update.effective_user.id if update.effective_user else None
    if uid is None:
        return
    if not _is_admin(cfg, uid):
        await msg.reply_text("Only the bot administrator may run /backtest_all.")
        return

    jobs = _job_coordinator(context)
    busy = await jobs.try_acquire_backtest_all(uid, cfg.project_root)
    if busy == "busy_other":
        await msg.reply_text(
            "A bulk backtest is already in progress (another user or session). "
            "Please try again when it has finished."
        )
        return
    if busy == "busy_self":
        await msg.reply_text(
            "A bulk backtest is already running for you. Please wait for it to finish before starting another."
        )
        return

    try:
        ensure_input_stocks(cfg.project_root)
        stock_filters = filters_for_subprocess(parse_filters_from_command_args(context.args or []))
        filt_line = _filter_summary_html(stock_filters)
        progress = await msg.reply_text(
            "<b>backtest_all</b>\n"
            + filt_line
            + "<i>Starting… (this single message will update)</i>",
            parse_mode="HTML",
        )
        chat_id, mid = progress.chat_id, progress.message_id
        bot = context.bot

        async def on_progress(full: str) -> None:
            body = html.escape(format_stream_blob(full))
            html_text = (
                "<b>backtest_all</b>\n"
                + filt_line
                + "<i>Running…</i>\n"
                f"<pre>{body}</pre>"
            )
            await _telegram_edit_stream_message(bot, chat_id, mid, html_text)

        try:
            code, log_text = await run_backtest_all_module_async(
                cfg.project_root,
                strat,
                on_progress=on_progress,
                mirror_terminal=True,
                min_interval=0.55,
                stock_filters=stock_filters,
            )
        except Exception as exc:
            log.exception("backtest_all failed")
            await _telegram_edit_stream_message(
                bot,
                chat_id,
                mid,
                f"<b>backtest_all</b>\n<pre>{html.escape(repr(exc))}</pre>",
            )
            return

        done_body = html.escape(format_stream_blob(log_text or ""))
        await _telegram_edit_stream_message(
            bot,
            chat_id,
            mid,
            "<b>backtest_all</b>\n"
            + filt_line
            + f"<i>Finished. Exit code: {code}</i>\n"
            f"<pre>{done_body}</pre>",
        )

        csv_path = backtest_all_csv_path(cfg.project_root, strat)
        loop = asyncio.get_running_loop()
        if csv_path.is_file():
            await loop.run_in_executor(None, functools.partial(_post_backtest_csv_merge, cfg, csv_path))

        if not csv_path.is_file():
            await msg.reply_text(f"No CSV at {csv_path} (see log in the message above).")
            return

        work = cfg.project_root / WORK_SUBDIR
        paths = prepare_csv_for_telegram(
            csv_path,
            max_bytes=cfg.max_upload_bytes,
            work_dir=work,
        )
        for i, p in enumerate(paths):
            cap = f"{csv_path.name} ({i + 1}/{len(paths)})" if len(paths) > 1 else csv_path.name
            data = p.read_bytes()
            await msg.reply_document(
                document=InputFile(data, filename=p.name),
                caption=cap[:1024],
            )
    finally:
        await jobs.release_backtest_all(cfg.project_root)


async def cmd_live_signals(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _ensure_allowed(update, context):
        return
    strat = await _ensure_strategy(update, context)
    if not strat:
        return
    msg = update.effective_message
    if not msg:
        return
    cfg = _cfg(context)
    uid = update.effective_user.id if update.effective_user else None
    if uid is None:
        return
    jobs = _job_coordinator(context)
    busy = await jobs.try_acquire_live_signals(uid, cfg.project_root)
    if busy == "busy_other":
        await msg.reply_text(
            "Live signals is currently being used by another user. "
            "Please try again after it has finished."
        )
        return
    if busy == "busy_self":
        await msg.reply_text(
            "Live signals is already running for you. Please wait for the current run to finish."
        )
        return
    try:
        ensure_input_stocks(cfg.project_root)
        parsed_filters = parse_filters_from_command_args(context.args or [])
        stock_filters = filters_for_subprocess(parsed_filters)
        if "fib" in parsed_filters:
            fib_on = parsed_filters.get("fib") == "true"
        else:
            fib_on = bool(uid is not None and _is_admin(cfg, uid))
        filt_line = _filter_summary_html(parsed_filters)
        if "fib" not in parsed_filters:
            if fib_on:
                filt_line += "<i>Fib columns: on (admin default; use fib=false to skip).</i>\n"
            else:
                filt_line += "<i>Fib columns: off (default; use fib=true to enable).</i>\n"
        signal_side = parsed_filters.get("signal")
        progress = await msg.reply_text(
            "<b>live_signals</b>\n"
            + filt_line
            + "<i>Starting… (this single message will update)</i>",
            parse_mode="HTML",
        )
        chat_id, mid = progress.chat_id, progress.message_id
        bot = context.bot

        async def on_progress(full: str) -> None:
            body = html.escape(format_stream_blob(full))
            html_text = (
                "<b>live_signals</b>\n"
                + filt_line
                + "<i>Running…</i>\n"
                f"<pre>{body}</pre>"
            )
            await _telegram_edit_stream_message(bot, chat_id, mid, html_text)

        try:
            code, log_text = await run_live_signals_module_async(
                cfg.project_root,
                strat,
                on_progress=on_progress,
                mirror_terminal=True,
                min_interval=0.55,
                stock_filters=stock_filters,
            )
        except Exception as exc:
            log.exception("live_signals failed")
            await _telegram_edit_stream_message(
                bot,
                chat_id,
                mid,
                f"<b>live_signals</b>\n<pre>{html.escape(repr(exc))}</pre>",
            )
            return

        done_body = html.escape(format_stream_blob(log_text or ""))
        await _telegram_edit_stream_message(
            bot,
            chat_id,
            mid,
            "<b>live_signals</b>\n"
            + filt_line
            + f"<i>Finished. Exit code: {code}</i>\n"
            f"<pre>{done_body}</pre>",
        )

        out = live_signals_csv_path(cfg.project_root, strat)
        if not out.is_file():
            await msg.reply_text("No output file produced (see log in the message above).")
            return
        work = cfg.project_root / WORK_SUBDIR
        loop = asyncio.get_running_loop()

        if fib_on and out.stat().st_size > 0:
            await _telegram_edit_stream_message(
                bot,
                chat_id,
                mid,
                "<b>live_signals</b>\n"
                + filt_line
                + f"<i>Finished (exit {code}). Adding Fib columns (1d / 15m / 5m)…</i>\n"
                f"<pre>{done_body}</pre>",
            )
            await loop.run_in_executor(None, enrich_live_signals_csv_with_fib, out)
            await _telegram_edit_stream_message(
                bot,
                chat_id,
                mid,
                "<b>live_signals</b>\n"
                + filt_line
                + f"<i>Finished. Exit code: {code}</i> — Fib columns attached.\n"
                f"<pre>{done_body}</pre>",
            )

        def _prepare_live_upload() -> tuple[list[Path], bool]:
            upload_csv, applied = materialize_live_signals_upload_path(out, work, signal_side)
            paths = prepare_csv_for_telegram(
                upload_csv,
                max_bytes=cfg.max_upload_bytes,
                work_dir=work,
            )
            return paths, applied

        paths, applied_side = await loop.run_in_executor(None, _prepare_live_upload)
        if not paths:
            await msg.reply_text("Live signals CSV is empty (nothing to send).")
            return
        fib_hint = " — Fib 1d/15m/5m" if fib_on else ""
        for i, p in enumerate(paths):
            if len(paths) > 1:
                part = f" ({i + 1}/{len(paths)})"
            else:
                part = ""
            if signal_side and applied_side:
                cap = f"{p.name}{part} — {signal_side} signals only{fib_hint}"
            elif signal_side and not applied_side:
                cap = (
                    f"{p.name}{part} — asked for {signal_side} only, "
                    "but CSV has no `signal` column; sending full file"
                )
                if fib_hint:
                    cap += fib_hint
            else:
                cap = (f"{p.name}{part}" if part else p.name) + fib_hint
            data = p.read_bytes()
            await msg.reply_document(
                document=InputFile(data, filename=p.name),
                caption=cap[:1024],
            )
    finally:
        await jobs.release_live_signals(cfg.project_root)


def _analyze_blocking(cfg: Config, strategy: str, symbol: str) -> SingleStockAnalysis:
    ensure_input_stocks(cfg.project_root)
    return analyze_single_stock(cfg.project_root, strategy, symbol)


def _deep_analyze_blocking(cfg: Config, strategy: str, symbol: str) -> SingleStockAnalysis:
    ensure_input_stocks(cfg.project_root)
    return analyze_single_stock_from_saved_params(cfg.project_root, strategy, symbol)


async def cmd_bt1(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _ensure_allowed(update, context):
        return
    strat = await _ensure_strategy(update, context)
    if not strat:
        return
    msg = update.effective_message
    if not msg:
        return
    if not context.args:
        await msg.reply_text("Usage: `/bt1 RELIANCE`", parse_mode="Markdown")
        return
    sym = context.args[0].strip().upper()
    cfg = _cfg(context)
    universe = load_symbol_universe(cfg.project_root)
    if sym not in universe:
        await msg.reply_text(f"`{sym}` is not in your input/STOCKS.csv universe.", parse_mode="Markdown")
        return

    await msg.reply_text(f"Optimizing `{sym}`…", parse_mode="Markdown")
    loop = asyncio.get_running_loop()
    try:
        res = await loop.run_in_executor(
            None,
            functools.partial(_analyze_blocking, cfg, strat, sym),
        )
    except Exception as exc:
        log.exception("bt1 failed")
        await msg.reply_text(f"Failed: {exc!r}")
        return

    if not res.ok:
        err = html.escape(res.error or "")
        await msg.reply_text(
            f"<b>{html.escape(sym)}</b>: failed — <pre>{err}</pre>",
            parse_mode="HTML",
        )
        return

    await _send_analysis_bundle(msg, res, sym)


async def cmd_deep(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _ensure_allowed(update, context):
        return
    strat = await _ensure_strategy(update, context)
    if not strat:
        return
    msg = update.effective_message
    if not msg:
        return
    if not context.args:
        await msg.reply_text("Usage: `/deep RELIANCE`", parse_mode="Markdown")
        return
    sym = context.args[0].strip().upper()
    cfg = _cfg(context)
    universe = load_symbol_universe(cfg.project_root)
    if sym not in universe:
        await msg.reply_text(f"`{sym}` is not in your input/STOCKS.csv universe.", parse_mode="Markdown")
        return

    await msg.reply_text(f"Deep dive `{sym}` (saved params)…", parse_mode="Markdown")
    loop = asyncio.get_running_loop()
    try:
        res = await loop.run_in_executor(
            None,
            functools.partial(_deep_analyze_blocking, cfg, strat, sym),
        )
    except Exception as exc:
        log.exception("deep failed")
        await msg.reply_text(f"Failed: {exc!r}")
        return

    if not res.ok:
        err = html.escape(res.error or "")
        await msg.reply_text(
            f"<b>{html.escape(sym)}</b>: failed — <pre>{err}</pre>",
            parse_mode="HTML",
        )
        return

    await _send_analysis_bundle(msg, res, sym)


def _blocking_fib(sym: str) -> str:
    pkg = compute_fib_package(sym)
    return fib_package_to_text(pkg)


def _blocking_screener(sym: str) -> str:
    return fetch_screener_bundle(sym)


async def cmd_fib(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Latest swing Fibonacci levels on daily, 15-minute, and 5-minute bars."""
    if not await _ensure_allowed(update, context):
        return
    msg = update.effective_message
    if not msg:
        return
    if not context.args:
        await msg.reply_text("Usage: `/fib RELIANCE`", parse_mode="Markdown")
        return
    sym = context.args[0].strip().upper()
    cfg = _cfg(context)
    universe = load_symbol_universe(cfg.project_root)
    if sym not in universe:
        await msg.reply_text(f"`{sym}` is not in your input/STOCKS.csv universe.", parse_mode="Markdown")
        return
    await msg.reply_text(f"Computing Fib swings for `{sym}`…", parse_mode="Markdown")
    loop = asyncio.get_running_loop()
    try:
        text = await loop.run_in_executor(None, functools.partial(_blocking_fib, sym))
    except Exception as exc:
        log.exception("fib failed")
        await msg.reply_text(f"Failed: {exc!r}")
        return
    body = f"<b>Fib swing</b> — <code>{html.escape(sym)}</code>\n<pre>{html.escape(text)}</pre>"
    await msg.reply_text(_clamp_telegram_html(body, 4096), parse_mode="HTML")


async def cmd_screener(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Screener.in: long automated read in chat (split messages if needed). No file attachment."""
    if not await _ensure_allowed(update, context):
        return
    msg = update.effective_message
    if not msg:
        return
    if not context.args:
        await msg.reply_text("Usage: `/screener RELIANCE`", parse_mode="Markdown")
        return
    sym = context.args[0].strip().upper()
    cfg = _cfg(context)
    universe = load_symbol_universe(cfg.project_root)
    if sym not in universe:
        await msg.reply_text(f"`{sym}` is not in your input/STOCKS.csv universe.", parse_mode="Markdown")
        return
    await msg.reply_text(
        f"Fetching screener.in data for `{sym}` (several HTTP calls; may take ~30–60s)…",
        parse_mode="Markdown",
    )
    loop = asyncio.get_running_loop()
    try:
        summary = await loop.run_in_executor(None, functools.partial(_blocking_screener, sym))
    except Exception as exc:
        log.exception("screener failed")
        await msg.reply_text(f"Failed: {exc!r}")
        return
    parts = _split_plain_text_for_pre(summary)
    n = len(parts)
    for i, part in enumerate(parts):
        if i == 0:
            hdr = (
                f"<b>Screener.in</b> — <code>{html.escape(sym)}</code>\n"
                "<i>Automated read from site tables — not financial advice. "
                "Verdict uses fixed rules, not a real AI model.</i>\n\n"
            )
        else:
            hdr = f"<i>Continued ({i + 1}/{n})</i>\n\n"
        body = hdr + f"<pre>{html.escape(part)}</pre>"
        await msg.reply_text(_clamp_telegram_html(body, 4096), parse_mode="HTML")


async def post_init(application: Application) -> None:
    cfg = application.bot_data["config"]
    clear_stale_job_busy_files(cfg.project_root)
    log.info(
        "Bot started; project_root=%s admin_user_id=%s",
        cfg.project_root,
        cfg.admin_user_id,
    )


def main() -> None:
    cfg = load_config()
    app = (
        Application.builder()
        .token(cfg.telegram_token)
        .post_init(post_init)
        # Default connect timeout is 5s; slow networks during long jobs caused ConnectTimeout on edits.
        .connect_timeout(25.0)
        .read_timeout(120.0)
        .write_timeout(60.0)
        .build()
    )
    app.bot_data["config"] = cfg
    app.bot_data["allowlist_lock"] = asyncio.Lock()
    app.bot_data["jobs"] = GlobalJobCoordinator()
    read_extra_user_ids(cfg.project_root)
    app.add_handler(
        MessageHandler(filters.TEXT & (~filters.COMMAND), _log_non_command_text, block=False),
        group=-1,
    )
    app.add_handler(CommandHandler("admin_add", cmd_admin_add))
    app.add_handler(CommandHandler("admin_remove", cmd_admin_remove))
    app.add_handler(CommandHandler("admin_list", cmd_admin_list))
    app.add_handler(CommandHandler("start", _log_command_lifecycle("start", cmd_start)))
    app.add_handler(CommandHandler("help", _log_command_lifecycle("help", cmd_help)))
    app.add_handler(CommandHandler("strategy", _log_command_lifecycle("strategy", cmd_strategy)))
    app.add_handler(CommandHandler("get_stocks", _log_command_lifecycle("get_stocks", cmd_get_stocks)))
    app.add_handler(CommandHandler("get_sectors", _log_command_lifecycle("get_sectors", cmd_get_sectors)))
    app.add_handler(CommandHandler("get_industries", _log_command_lifecycle("get_industries", cmd_get_industries)))
    app.add_handler(
        CallbackQueryHandler(on_strategy_callback, pattern=r"^strategy:(sma|ema)$")
    )
    app.add_handler(CommandHandler("backtest_all", _log_command_lifecycle("backtest_all", cmd_backtest_all)))
    app.add_handler(CommandHandler("live_signals", _log_command_lifecycle("live_signals", cmd_live_signals)))
    app.add_handler(CommandHandler("bt1", _log_command_lifecycle("bt1", cmd_bt1)))
    app.add_handler(CommandHandler("deep", _log_command_lifecycle("deep", cmd_deep)))
    app.add_handler(CommandHandler("fib", _log_command_lifecycle("fib", cmd_fib)))
    app.add_handler(CommandHandler("screener", _log_command_lifecycle("screener", cmd_screener)))
    log.info("Starting polling…")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
