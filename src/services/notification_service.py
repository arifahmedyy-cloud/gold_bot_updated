"""Notification service for Telegram and Discord alerts.

Sends structured alerts on trade events, errors, drawdowns, and disconnections.
"""

from __future__ import annotations

import requests
from typing import Optional
from datetime import datetime

from src.logger import get_logger
from src.config import NotificationConfig

log = get_logger(__name__)


class NotificationService:
    """Sends alerts via Telegram and/or Discord."""

    def __init__(self, config: NotificationConfig) -> None:
        self.config = config

    def _send_telegram(self, message: str) -> bool:
        if not self.config.telegram_bot_token or not self.config.telegram_chat_id:
            return False
        url = f"https://api.telegram.org/bot{self.config.telegram_bot_token}/sendMessage"
        try:
            resp = requests.post(
                url, json={"chat_id": self.config.telegram_chat_id, "text": message, "parse_mode": "Markdown"},
                timeout=10
            )
            return resp.status_code == 200
        except Exception as exc:
            log.warning("Telegram send failed: %s", exc)
            return False

    def _send_discord(self, message: str) -> bool:
        if not self.config.discord_webhook_url:
            return False
        try:
            resp = requests.post(
                self.config.discord_webhook_url, json={"content": message}, timeout=10
            )
            return resp.status_code in (200, 204)
        except Exception as exc:
            log.warning("Discord send failed: %s", exc)
            return False

    def send(self, message: str) -> None:
        """Broadcast message to all configured channels."""
        if not self.config.enabled:
            return
        log.info("Notification: %s", message[:200])
        self._send_telegram(message)
        self._send_discord(message)

    def notify_trade(self, direction: str, entry: float, sl: float, tp: float, lot: float) -> None:
        msg = f"🚀 *Trade Executed*\nDirection: {direction}\nEntry: {entry}\nSL: {sl}\nTP: {tp}\nLot: {lot}"
        self.send(msg)

    def notify_error(self, error_message: str) -> None:
        msg = f"⚠️ *Bot Error*\n{error_message[:500]}"
        self.send(msg)

    def notify_drawdown(self, dd_pct: float, threshold: float) -> None:
        msg = f"🚨 *Drawdown Alert*\nCurrent DD: {dd_pct:.2f}% (threshold: {threshold:.2f}%)"
        self.send(msg)

    def notify_disconnect(self, broker_name: str) -> None:
        msg = f"🔌 *Broker Disconnected*\n{broker_name} connection lost at {datetime.now().isoformat()}"
        self.send(msg)
