"""Serialize heavy Telegram-triggered jobs so concurrent users do not share one subprocess stream."""
from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger("trade_bot.global_jobs")

LIVE_SIGNALS_BUSY = Path("_telegram_work") / "live_signals_busy.json"
BACKTEST_ALL_BUSY = Path("_telegram_work") / "backtest_all_busy.json"


def _busy_path(project_root: Path, rel: Path) -> Path:
    return project_root / rel


def clear_stale_job_busy_files(project_root: Path) -> None:
    """
    Remove on-disk busy markers from a previous process (crash / kill / restart).

    A new bot process always clears these so jobs are not stuck "running" forever.
    """
    for rel in (LIVE_SIGNALS_BUSY, BACKTEST_ALL_BUSY):
        p = _busy_path(project_root, rel)
        try:
            p.unlink(missing_ok=True)
        except OSError as exc:
            log.warning("could not remove stale busy file %s: %s", p, exc)


def _write_busy_file(path: Path, user_id: int, job: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "pid": os.getpid(),
        "user_id": int(user_id),
        "job": job,
        "started": datetime.now(timezone.utc).isoformat(),
    }
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def _unlink_busy_file(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        log.warning("could not remove busy file %s: %s", path, exc)


class GlobalJobCoordinator:
    """
    At most one ``live_signals`` and one ``backtest_all`` run at a time (each global).

    Busy markers on disk are best-effort hints; the in-process guard is authoritative.
    """

    __slots__ = ("_mutex", "_live_signals_uid", "_backtest_all_uid")

    def __init__(self) -> None:
        self._mutex = asyncio.Lock()
        self._live_signals_uid: int | None = None
        self._backtest_all_uid: int | None = None

    async def try_acquire_live_signals(self, user_id: int, project_root: Path) -> str:
        """Return ``ok`` | ``busy_self`` | ``busy_other``."""
        async with self._mutex:
            if self._live_signals_uid is not None:
                return "busy_self" if self._live_signals_uid == int(user_id) else "busy_other"
            self._live_signals_uid = int(user_id)
            _write_busy_file(_busy_path(project_root, LIVE_SIGNALS_BUSY), user_id, "live_signals")
            return "ok"

    async def release_live_signals(self, project_root: Path) -> None:
        async with self._mutex:
            self._live_signals_uid = None
            _unlink_busy_file(_busy_path(project_root, LIVE_SIGNALS_BUSY))

    async def try_acquire_backtest_all(self, user_id: int, project_root: Path) -> str:
        """Return ``ok`` | ``busy_self`` | ``busy_other``."""
        async with self._mutex:
            if self._backtest_all_uid is not None:
                return "busy_self" if self._backtest_all_uid == int(user_id) else "busy_other"
            self._backtest_all_uid = int(user_id)
            _write_busy_file(_busy_path(project_root, BACKTEST_ALL_BUSY), user_id, "backtest_all")
            return "ok"

    async def release_backtest_all(self, project_root: Path) -> None:
        async with self._mutex:
            self._backtest_all_uid = None
            _unlink_busy_file(_busy_path(project_root, BACKTEST_ALL_BUSY))
