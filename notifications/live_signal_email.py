"""
Optional SMTP email when a live-signals CSV contains one or more rows.

**Do not put passwords in code or git.** Use a Gmail *App Password* (2FA required) and set
environment variables, or inject them in your scheduler / OS user profile.

Enable mail when live screen finds signals (non-empty CSV):

- Set ``EMAIL_ON_LIVE_SIGNALS = True`` in ``Algorithms/.../settings.py`` **and**
- Export the following (names are fixed so both SMA and EMA use the same mailbox):

  ``TRADE_PILOT_SMTP_HOST`` — default ``smtp.gmail.com``  
  ``TRADE_PILOT_SMTP_PORT`` — default ``587``  
  ``TRADE_PILOT_SMTP_USER`` — Gmail address used to authenticate  
  ``TRADE_PILOT_SMTP_PASSWORD`` — app password (not your normal Gmail password)  
  ``TRADE_PILOT_EMAIL_TO`` — comma-separated recipient addresses  
  ``TRADE_PILOT_EMAIL_FROM`` — optional; defaults to ``TRADE_PILOT_SMTP_USER``

If any required variable is missing, the function logs a short message to stderr and returns.
"""

from __future__ import annotations

import os
import smtplib
import sys
import warnings
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

import pandas as pd


def mail_live_signals_csv_if_nonempty(
    *,
    csv_path: Path,
    strategy_label: str,
    subject_prefix: str = "Trade Pilot",
) -> None:
    """
    If ``csv_path`` exists, is readable, and has at least one data row, send it as an attachment.

    Credentials come **only** from environment variables (see module docstring).
    """
    if not csv_path.is_file():
        return
    try:
        df = pd.read_csv(csv_path)
    except (OSError, pd.errors.EmptyDataError, ValueError) as e:
        warnings.warn(f"live_signal_email: skip mail, could not read CSV: {e}", stacklevel=2)
        return
    if df.empty:
        return

    host = os.environ.get("TRADE_PILOT_SMTP_HOST", "smtp.gmail.com").strip()
    port_s = os.environ.get("TRADE_PILOT_SMTP_PORT", "587").strip()
    user = os.environ.get("TRADE_PILOT_SMTP_USER", "bhuvanesh.valiveti@gmail.com").strip()
    password = os.environ.get("TRADE_PILOT_SMTP_PASSWORD", "bcea clul fryl htqw").strip()
    to_raw = os.environ.get("TRADE_PILOT_EMAIL_TO", "bhuvanesh.valiveti@gmail.com").strip()
    mail_from = os.environ.get("TRADE_PILOT_EMAIL_FROM", user).strip()

    try:
        port = int(port_s)
    except ValueError:
        print("live_signal_email: TRADE_PILOT_SMTP_PORT invalid", file=sys.stderr)
        return

    if not user or not password or not to_raw:
        print(
            "live_signal_email: missing TRADE_PILOT_SMTP_USER, TRADE_PILOT_SMTP_PASSWORD, "
            "or TRADE_PILOT_EMAIL_TO — not sending mail.",
            file=sys.stderr,
        )
        return

    receivers = [x.strip() for x in to_raw.split(",") if x.strip()]
    if not receivers:
        print("live_signal_email: TRADE_PILOT_EMAIL_TO has no addresses", file=sys.stderr)
        return

    n = len(df)
    subject = f"{subject_prefix}: {strategy_label} — {n} signal(s)"
    body = (
        f"The live screen for **{strategy_label}** found **{n}** row(s) with a buy/sell signal "
        f"on the latest bar.\n\n"
        f"CSV attached: {csv_path.name}\n"
    )

    msg = MIMEMultipart()
    msg["Subject"] = subject
    msg["From"] = mail_from
    msg["To"] = ", ".join(receivers)
    msg.attach(MIMEText(body, "plain", "utf-8"))

    data = csv_path.read_bytes()
    part = MIMEApplication(data, Name=csv_path.name)
    part.add_header("Content-Disposition", "attachment", filename=csv_path.name)
    msg.attach(part)

    try:
        with smtplib.SMTP(host, port, timeout=60) as server:
            server.starttls()
            server.login(user, password)
            server.send_message(msg)
    except OSError as e:
        print(f"live_signal_email: SMTP failed: {e}", file=sys.stderr)
