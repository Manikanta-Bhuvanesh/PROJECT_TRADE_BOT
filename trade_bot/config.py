from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

_TRADE_BOT_PKG = Path(__file__).resolve().parent
# Project root: folder that contains ``trade_bot/``, ``Algorithms/``, ``input/``, …
_PROJECT_ROOT = _TRADE_BOT_PKG.parent

DEFAULT_ADMIN_USER_ID = 945784800


def _parse_allowed_ids(raw: str | None) -> set[int]:
    if not raw or not str(raw).strip():
        return set()
    out: set[int] = set()
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.add(int(part))
        except ValueError:
            continue
    return out


def _parse_admin_user_id(raw: str | None) -> int:
    """Telegram user id that may manage the bot-managed allowlist; defaults to project owner."""
    if raw is None or not str(raw).strip():
        return DEFAULT_ADMIN_USER_ID
    try:
        v = int(str(raw).strip())
        if v <= 0:
            return DEFAULT_ADMIN_USER_ID
        return v
    except ValueError:
        return DEFAULT_ADMIN_USER_ID


@dataclass(frozen=True)
class Config:
    telegram_token: str
    """Root of this repo: ``Algorithms/``, ``data_fetcher/``, ``input/``, ``output/``."""
    project_root: Path
    allowed_user_ids: set[int]
    """Telegram user id with full access and hidden allowlist commands (see ``ADMIN_USER_ID`` env)."""
    admin_user_id: int
    max_upload_bytes: int = 48 * 1024 * 1024

    def is_user_allowed(self, user_id: int | None) -> bool:
        from trade_bot.services.allowlist_store import read_extra_user_ids

        if user_id is None:
            return False
        uid = int(user_id)
        if uid == int(self.admin_user_id):
            return True
        env_ids = self.allowed_user_ids
        extra = read_extra_user_ids(self.project_root)
        if not env_ids and not extra:
            return True
        return uid in env_ids | extra


def load_config() -> Config:
    load_dotenv(_PROJECT_ROOT / ".env")
    token = (os.environ.get("TELEGRAM_BOT_TOKEN") or "").strip()
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN missing. Copy .env.example to .env and set the token.")

    project_root = _PROJECT_ROOT.resolve()
    algo = project_root / "Algorithms" / "brute_sma_cross" / "settings.py"
    if not algo.is_file():
        raise RuntimeError(
            f"Vendored engine missing under {project_root} (expected Algorithms/brute_sma_cross/). "
            "This project is self-contained and does not use an external screener path."
        )

    allowed = _parse_allowed_ids(os.environ.get("ALLOWED_USER_IDS"))
    admin = _parse_admin_user_id(os.environ.get("ADMIN_USER_ID"))
    return Config(
        telegram_token=token,
        project_root=project_root,
        allowed_user_ids=allowed,
        admin_user_id=admin,
    )
