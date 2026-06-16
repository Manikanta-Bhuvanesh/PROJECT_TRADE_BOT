"""Optional user notifications (e.g. live-signal email)."""

from .live_signal_email import mail_live_signals_csv_if_nonempty

__all__ = ["mail_live_signals_csv_if_nonempty"]
