"""Persistent extra Telegram user IDs (admin-managed); not referenced from /help."""
from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger("trade_bot.allowlist")

# Same work dir as CSV splits; filename is explicit for .gitignore.
ALLOWLIST_REL_PATH = Path("_telegram_work") / "allowed_user_ids.json"

_cache_mtime: float | None = None
_cache_ids: frozenset[int] | None = None


def allowlist_path(project_root: Path) -> Path:
    return project_root / ALLOWLIST_REL_PATH


def _invalidate_cache() -> None:
    global _cache_mtime, _cache_ids
    _cache_mtime = None
    _cache_ids = None


def read_extra_user_ids(project_root: Path) -> set[int]:
    """User IDs stored by the admin via bot commands (unioned with env allowlist)."""
    global _cache_mtime, _cache_ids
    path = allowlist_path(project_root)
    if not path.is_file():
        _cache_mtime = None
        _cache_ids = frozenset()
        return set()
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return set()
    if _cache_mtime == mtime and _cache_ids is not None:
        return set(_cache_ids)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        ids = raw.get("user_ids") if isinstance(raw, dict) else None
        if not isinstance(ids, list):
            out: set[int] = set()
        else:
            out = set()
            for x in ids:
                try:
                    out.add(int(x))
                except (TypeError, ValueError):
                    continue
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("allowlist read failed %s: %s", path, exc)
        out = set()
    _cache_mtime = mtime
    _cache_ids = frozenset(out)
    return set(out)


def write_extra_user_ids(project_root: Path, user_ids: set[int]) -> None:
    """Replace stored extra IDs (atomic replace)."""
    path = allowlist_path(project_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({"user_ids": sorted(user_ids)}, indent=2, ensure_ascii=False)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(payload + "\n", encoding="utf-8")
    tmp.replace(path)
    _invalidate_cache()


def warm_cache(project_root: Path) -> None:
    """Load allowlist once at startup (optional)."""
    read_extra_user_ids(project_root)
