"""Health monitoring with notifications and auto-reconnect.

Wraps any BrokerConnector and tracks system health metrics.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime
from typing import Any, Callable, Dict, Optional

from src.logger import get_logger
from src.models import HealthStatus
from src.services.notification_service import NotificationService

log = get_logger(__name__)


class HealthService:
    """Monitors broker health and system status."""

    def __init__(
        self,
        broker,
        interval_seconds: int = 30,
        on_unhealthy: Optional[Callable[[Dict], None]] = None,
        auto_reconnect: bool = True,
        notifier: Optional[NotificationService] = None,
    ) -> None:
        self.broker = broker
        self.interval_seconds = interval_seconds
        self.on_unhealthy = on_unhealthy
        self.auto_reconnect = auto_reconnect
        self.notifier = notifier

        self._started_at = datetime.now()
        self._status: Dict[str, Any] = {"connected": False}
        self._last_market_data_update: Optional[datetime] = None
        self._last_successful_trade: Optional[Dict[str, Any]] = None
        self._last_error: Optional[Dict[str, Any]] = None
        self._last_order: Optional[Dict[str, Any]] = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def check_once(self) -> Dict[str, Any]:
        """Run a single health check synchronously."""
        try:
            status = self.broker.health_check()
        except Exception as exc:
            log.error("Health check failed: %s", exc)
            status = {"connected": False, "error": str(exc)}

        with self._lock:
            self._status = status

        if not status.get("connected", False):
            log.warning("Broker unhealthy: %s", status)
            if self.notifier:
                self.notifier.notify_disconnect(type(self.broker).__name__)
            if self.on_unhealthy:
                try:
                    self.on_unhealthy(status)
                except Exception as exc:
                    log.error("on_unhealthy callback error: %s", exc)
            if self.auto_reconnect and hasattr(self.broker, "connect_with_retry"):
                self.broker.connect_with_retry()
        return status

    def record_market_data_update(self, when: Optional[datetime] = None) -> None:
        with self._lock:
            self._last_market_data_update = when or datetime.now()

    def record_trade(self, profit_loss: float = 0.0, when: Optional[datetime] = None) -> None:
        with self._lock:
            self._last_successful_trade = {
                "time": when or datetime.now(),
                "profit_loss": profit_loss,
            }

    def record_order(
        self, action: str, symbol: str, success: bool,
        ticket: Optional[int] = None, reason: Optional[str] = None,
        when: Optional[datetime] = None,
    ) -> None:
        with self._lock:
            self._last_order = {
                "time": when or datetime.now(),
                "action": action, "symbol": symbol,
                "success": success, "ticket": ticket, "reason": reason,
            }

    def record_error(self, message: str, when: Optional[datetime] = None) -> None:
        with self._lock:
            self._last_error = {
                "time": when or datetime.now(),
                "message": str(message),
            }
        if self.notifier and "daily loss limit" in str(message).lower():
            self.notifier.send(f"🛑 *Risk Alert*\n{message}")

    def latest_status(self) -> HealthStatus:
        with self._lock:
            uptime = (datetime.now() - self._started_at).total_seconds()
            return HealthStatus(
                connected=self._status.get("connected", False),
                symbol=self._status.get("symbol"),
                terminal_ok=self._status.get("terminal_ok", False),
                account_ok=self._status.get("account_ok", False),
                checked_at=self._status.get("checked_at", datetime.now()),
                uptime_seconds=uptime,
                last_market_data_update=self._last_market_data_update,
                last_successful_trade=self._last_successful_trade,
                last_error=self._last_error,
                last_order=self._last_order,
            )

    def _run(self) -> None:
        while not self._stop_event.is_set():
            self.check_once()
            self._stop_event.wait(self.interval_seconds)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        log.info("Health monitor started (interval=%ds)", self.interval_seconds)

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        log.info("Health monitor stopped")
