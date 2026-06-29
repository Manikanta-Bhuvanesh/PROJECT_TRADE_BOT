"""Run ``python -m <module>`` from ``project_root`` with merged stdout/stderr streaming."""
from __future__ import annotations

import asyncio
import logging
import os
import re
import sys
from collections.abc import Awaitable, Callable
from pathlib import Path

_log = logging.getLogger(__name__)

# Strip ANSI so Telegram <pre> stays readable
ANSI_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


def collapse_carriage_returns(s: str) -> str:
    """
    Apply ``\\r`` semantics so tqdm-style updates become one line instead of many.

    tqdm refreshes with ``\\r`` (same line). Replacing ``\\r`` with ``\\n`` stacks every
    snapshot; collapsing keeps only the latest bar state per logical line.
    """
    lines: list[str] = []
    cur: list[str] = []
    for ch in s:
        if ch == "\r":
            cur.clear()
        elif ch == "\n":
            lines.append("".join(cur))
            cur.clear()
        else:
            cur.append(ch)
    tail = "".join(cur)
    if not lines:
        return tail
    if tail:
        return "\n".join(lines) + "\n" + tail
    return "\n".join(lines)


def format_stream_blob(blob: str, *, max_inner: int = 3600) -> str:
    """Normalize captured subprocess output for a Telegram message body."""
    s = ANSI_RE.sub("", blob)
    s = collapse_carriage_returns(s)
    if len(s) > max_inner:
        s = "…\n" + s[-max_inner:]
    return s


async def run_python_module_streaming(
    project_root: Path,
    module: str,
    *,
    on_progress: Callable[[str], Awaitable[None]] | None = None,
    mirror_terminal: bool = True,
    min_interval: float = 0.65,
    extra_env: dict[str, str] | None = None,
) -> tuple[int, str]:
    """
    Run ``sys.executable -m module`` with cwd ``project_root``.

    Merged stdout/stderr is accumulated. If ``on_progress`` is set, it receives the
    full log-so-far on a throttled schedule while the process runs, then once
    more at the end (so the last snapshot is always flushed).

    Returns ``(exit_code, full_log_text)``.
    """
    cmd = [sys.executable, "-m", module]
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("TQDM_ASCII", "1")
    if extra_env:
        env.update(extra_env)
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=str(project_root.resolve()),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        stdin=asyncio.subprocess.DEVNULL,
        env=env,
    )
    if proc.stdout is None:
        return int(await proc.wait() or 0), ""

    chunks: list[str] = []
    loop = asyncio.get_running_loop()
    last_progress = 0.0
    progress_emitted = False

    async def emit(full: str, *, force: bool) -> None:
        nonlocal last_progress, progress_emitted
        if on_progress is None:
            return
        now = loop.time()
        # First snapshot is always sent so Telegram never looks blank until the throttle window.
        if not force and progress_emitted and (now - last_progress) < min_interval:
            return
        progress_emitted = True
        last_progress = now
        try:
            await on_progress(full)
        except Exception as exc:
            # e.g. httpx/httpcore timeouts while editing Telegram — do not abort the subprocess.
            _log.warning("on_progress failed (continuing job): %s", exc)

    while True:
        block = await proc.stdout.read(4096)
        if not block:
            break
        text = block.decode("utf-8", errors="replace")
        chunks.append(text)
        if mirror_terminal:
            sys.stderr.write(text)
            sys.stderr.flush()
        await emit("".join(chunks), force=False)

    code = int(await proc.wait() or 0)
    full = "".join(chunks)
    await emit(full, force=True)
    return code, full
