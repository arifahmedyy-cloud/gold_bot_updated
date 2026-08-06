"""Broker connectors with unified interface.

Supports Paper Trading and MetaTrader 5 with auto-reconnect,
symbol resolution, and correct margin calculations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any
from datetime import datetime, timedelta
from enum import Enum
import time
import math

import pandas as pd
import numpy as np

from src.logger import get_logger
from src.models import TradeSignal, Position, AccountInfo
from src.exceptions import BrokerError, ConnectionError, OrderError
from src.trading.symbol_manager import SymbolManager

log = get_logger(__name__)

_DEFAULT_SYMBOL_PRICES: Dict[str, Dict[str, float]] = {
    "XAUUSD": {"bid": 2450.0, "ask": 2450.5},
    "EURUSD": {"bid": 1.0850, "ask": 1.0852},
    "GBPUSD": {"bid": 1.2650, "ask": 1.2653},
    "USDJPY": {"bid": 149.50, "ask": 149.52},
    "AUDUSD": {"bid": 0.6550, "ask": 0.6553},
}


def _contract_size(symbol: str) -> float:
    """Units per 1.0 lot. Gold uses 100 troy ounces; standard forex lots
    are 100,000 units of base currency. Using the wrong size silently
    produces nonsensical P&L/margin for whichever asset class doesn't
    match, so every P&L calculation in PaperBroker must go through this."""
    profile = SymbolManager().profile(symbol)
    return 100.0 if profile.category == "gold" else 100_000.0


class BrokerConnector(ABC):
    """Abstract base class for all broker connectors."""

    @abstractmethod
    def connect(self) -> bool:
        """Establish connection to broker."""
        pass

    @abstractmethod
    def disconnect(self) -> bool:
        """Close connection."""
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """Check if connected."""
        pass

    @abstractmethod
    def get_account_info(self) -> AccountInfo:
        """Get account snapshot."""
        pass

    @abstractmethod
    def get_price(self, symbol: str) -> Dict[str, float]:
        """Get current bid/ask for symbol."""
        pass

    @abstractmethod
    def send_order(self, signal: TradeSignal) -> Dict[str, Any]:
        """Execute a trade order."""
        pass

    @abstractmethod
    def get_positions(self) -> List[Position]:
        """Get all open positions."""
        pass

    @abstractmethod
    def close_position(self, ticket: int) -> bool:
        """Close a specific position by ticket."""
        pass

    @abstractmethod
    def close_all_positions(self) -> bool:
        """Close all open positions."""
        pass

    def modify_position_sl_tp(
        self, ticket: int, sl: Optional[float] = None, tp: Optional[float] = None
    ) -> bool:
        """Modify SL and/or TP on an open position."""
        return False

    def close_position_partial(self, ticket: int, volume: float) -> Dict[str, Any]:
        """Close part of an open position."""
        return {"success": False, "error": "Partial close not supported"}

    def health_check(self) -> Dict[str, Any]:
        return {
            "connected": self.is_connected(),
            "symbol": None,
            "terminal_ok": self.is_connected(),
            "account_ok": self.is_connected(),
            "checked_at": datetime.now(),
        }


class PaperBroker(BrokerConnector):
    """Paper trading broker with realistic margin and lot sizing.

    Simulates gold (XAUUSD) trading where:
    - 1.0 lot = 100 troy ounces
    - P&L per lot per $1 move = $100
    - Margin uses configurable leverage (default 1:100)
    """

    def __init__(
        self,
        initial_balance: float = 10000.0,
        spread: float = 0.5,
        leverage: float = 100.0,
        symbol: str = "XAUUSD",
    ) -> None:
        self._balance = initial_balance
        self._equity = initial_balance
        self._spread = spread
        self._leverage = leverage
        self._symbol = symbol
        self._positions: List[Position] = []
        self._trade_history: List[Dict[str, Any]] = []
        self._ticket_counter = 1000
        self._connected = False
        self._symbol_prices: Dict[str, Dict[str, float]] = dict(_DEFAULT_SYMBOL_PRICES)
        if symbol not in self._symbol_prices:
            self._symbol_prices[symbol] = {"bid": 2450.0, "ask": 2450.5}

    def connect(self) -> bool:
        self._connected = True
        log.info("[PaperBroker] Connected (balance=$%.2f, leverage=1:%.0f)", self._balance, self._leverage)
        return True

    def disconnect(self) -> bool:
        self._connected = False
        return True

    def is_connected(self) -> bool:
        return self._connected

    def _notional(self, volume: float, price: float, symbol: Optional[str] = None) -> float:
        """Calculate notional value: volume * price * contract size (per-symbol)."""
        return volume * price * _contract_size(symbol or self._symbol)

    def _margin_required(self, volume: float, price: float, symbol: Optional[str] = None) -> float:
        """Calculate margin: notional / leverage."""
        return self._notional(volume, price, symbol) / self._leverage

    def get_account_info(self) -> AccountInfo:
        used_margin = sum(
            self._margin_required(p.volume, p.open_price, p.symbol) for p in self._positions
        )
        return AccountInfo(
            balance=self._balance,
            equity=self._equity,
            margin=used_margin,
            free_margin=self._equity - used_margin,
            margin_level=(self._equity / used_margin * 100) if used_margin > 0 else 100.0,
            leverage=self._leverage,
        )

    def get_price(self, symbol: str) -> Dict[str, float]:
        if symbol in self._symbol_prices:
            return self._symbol_prices[symbol]
        if symbol in _DEFAULT_SYMBOL_PRICES:
            return _DEFAULT_SYMBOL_PRICES[symbol]
        log.warning("[PaperBroker] No simulated price for %s, using a generic forex placeholder", symbol)
        return {"bid": 1.0000, "ask": 1.0002}

    def update_price(self, symbol: str, bid: float, ask: float) -> None:
        """Update simulated price and floating P&L."""
        self._symbol_prices[symbol] = {"bid": bid, "ask": ask}
        for pos in self._positions:
            if pos.symbol != symbol:
                continue
            if pos.position_type == "BUY":
                pos.profit = (bid - pos.open_price) * pos.volume * _contract_size(pos.symbol)
            else:
                pos.profit = (pos.open_price - ask) * pos.volume * _contract_size(pos.symbol)
        self._update_equity()

    def _update_equity(self) -> None:
        floating = sum(p.profit for p in self._positions)
        self._equity = self._balance + floating

    def send_order(self, signal: TradeSignal) -> Dict[str, Any]:
        if not self._connected:
            return {"success": False, "error": "Not connected"}

        self._ticket_counter += 1
        price = self.get_price(signal.symbol)
        fill_price = price["ask"] if signal.direction == "BUY" else price["bid"]

        margin = self._margin_required(signal.lot_size, fill_price, signal.symbol)
        if margin > self._equity:
            return {"success": False, "error": "Insufficient margin"}

        position = Position(
            ticket=self._ticket_counter,
            symbol=signal.symbol,
            position_type=signal.direction,
            volume=signal.lot_size,
            open_price=fill_price,
            sl=signal.sl,
            tp=signal.tp,
            open_time=signal.timestamp,
        )
        self._positions.append(position)
        self._trade_history.append({
            "ticket": self._ticket_counter,
            "type": "OPEN",
            "direction": signal.direction,
            "symbol": signal.symbol,
            "price": fill_price,
            "lot_size": signal.lot_size,
            "sl": signal.sl,
            "tp": signal.tp,
            "time": signal.timestamp,
        })
        log.info("[PaperBroker] OPEN #%d %s %.2f lots @ %.2f",
                 self._ticket_counter, signal.direction, signal.lot_size, fill_price)
        return {
            "success": True,
            "ticket": self._ticket_counter,
            "fill_price": fill_price,
            "message": f"{signal.direction} order executed at {fill_price}",
        }

    def get_positions(self) -> List[Position]:
        return self._positions.copy()

    def close_position(self, ticket: int) -> bool:
        for i, pos in enumerate(self._positions):
            if pos.ticket == ticket:
                price = self.get_price(pos.symbol)
                close_price = price["bid"] if pos.position_type == "BUY" else price["ask"]

                if pos.position_type == "BUY":
                    pl = (close_price - pos.open_price) * pos.volume * _contract_size(pos.symbol)
                else:
                    pl = (pos.open_price - close_price) * pos.volume * _contract_size(pos.symbol)

                self._balance += pl
                self._trade_history.append({
                    "ticket": ticket,
                    "type": "CLOSE",
                    "direction": pos.position_type,
                    "symbol": pos.symbol,
                    "open_price": pos.open_price,
                    "close_price": close_price,
                    "pl": pl,
                    "time": datetime.now(),
                })
                self._positions.pop(i)
                self._update_equity()
                log.info("[PaperBroker] CLOSE #%d P/L=$%.2f", ticket, pl)
                return True
        return False

    def close_all_positions(self) -> bool:
        tickets = [p.ticket for p in self._positions]
        results = [self.close_position(t) for t in tickets]
        return all(results)

    def modify_position_sl_tp(
        self, ticket: int, sl: Optional[float] = None, tp: Optional[float] = None
    ) -> bool:
        for pos in self._positions:
            if pos.ticket == ticket:
                if sl is not None:
                    pos.sl = sl
                if tp is not None:
                    pos.tp = tp
                log.info("[PaperBroker] MODIFY #%d SL=%s TP=%s", ticket, sl, tp)
                return True
        return False

    def close_position_partial(self, ticket: int, volume: float) -> Dict[str, Any]:
        for pos in self._positions:
            if pos.ticket == ticket:
                if volume <= 0 or volume >= pos.volume:
                    success = self.close_position(ticket)
                    return {"success": success, "closed_volume": pos.volume, "remaining_volume": 0.0}

                price = self.get_price(pos.symbol)
                close_price = price["bid"] if pos.position_type == "BUY" else price["ask"]
                if pos.position_type == "BUY":
                    pl = (close_price - pos.open_price) * volume * _contract_size(pos.symbol)
                else:
                    pl = (pos.open_price - close_price) * volume * _contract_size(pos.symbol)

                self._balance += pl
                pos.volume = round(pos.volume - volume, 4)
                self._trade_history.append({
                    "ticket": ticket, "type": "PARTIAL_CLOSE",
                    "direction": pos.position_type, "symbol": pos.symbol,
                    "open_price": pos.open_price, "close_price": close_price,
                    "volume_closed": volume, "pl": pl, "time": datetime.now(),
                })
                log.info("[PaperBroker] PARTIAL #%d closed %.2f lots P/L=$%.2f", ticket, volume, pl)
                return {"success": True, "closed_volume": volume, "remaining_volume": pos.volume}
        return {"success": False, "error": "Position not found"}

    def get_trade_history(self) -> List[Dict[str, Any]]:
        return self._trade_history

    def check_sl_tp(self, symbol: str, bid: float, ask: float) -> None:
        """Check if any positions hit SL or TP."""
        to_close = []
        for pos in self._positions:
            if pos.symbol != symbol:
                continue
            if pos.position_type == "BUY":
                if bid <= pos.sl:
                    to_close.append((pos.ticket, "SL"))
                elif bid >= pos.tp:
                    to_close.append((pos.ticket, "TP"))
            else:
                if ask >= pos.sl:
                    to_close.append((pos.ticket, "SL"))
                elif ask <= pos.tp:
                    to_close.append((pos.ticket, "TP"))
        for ticket, reason in to_close:
            self.close_position(ticket)
            log.info("[PaperBroker] Position #%d closed by %s", ticket, reason)


class MT5Broker(BrokerConnector):
    """MetaTrader 5 broker connector with auto-reconnect and symbol resolution."""

    def __init__(
        self,
        login: int = 0,
        password: str = "",
        server: str = "",
        symbol_candidates: Tuple[str, ...] = ("XAUUSD", "XAUUSDm", "XAUUSD.a", "XAUUSD.raw", "GOLD"),
        reconnect_attempts: int = 5,
        reconnect_backoff_seconds: float = 5.0,
        reconnect_backoff_multiplier: float = 2.0,
        leverage: float = 100.0,
    ) -> None:
        self.login = login
        self.password = password
        self.server = server
        self.symbol_candidates = symbol_candidates
        self.reconnect_attempts = reconnect_attempts
        self.reconnect_backoff_seconds = reconnect_backoff_seconds
        self.reconnect_backoff_multiplier = reconnect_backoff_multiplier
        self.leverage = leverage

        self._mt5 = None
        self._connected = False
        self.resolved_symbol: Optional[str] = None
        self._last_health_check: Optional[Dict[str, Any]] = None
        self._trade_history: List[Dict[str, Any]] = []
        self._position_cache: Dict[int, Position] = {}
        self._closed_position_ids: set = set()
        self._last_deal_sync: Optional[datetime] = None
        self.magic_number: int = 234000

    def connect(self) -> bool:
        try:
            import MetaTrader5 as mt5
            self._mt5 = mt5
            if not mt5.initialize():
                log.error("MT5 initialize failed: %s", mt5.last_error())
                return False
            if self.login > 0:
                authorized = mt5.login(self.login, password=self.password, server=self.server)
                if not authorized:
                    log.error("MT5 login failed: %s", mt5.last_error())
                    mt5.shutdown()
                    return False
            self._connected = True
            info = mt5.account_info()
            if info:
                log.info("MT5 connected. Balance=$%.2f Server=%s", info.balance, self.server)
            self.resolved_symbol = self.resolve_symbol()
            if self.resolved_symbol:
                log.info("MT5 resolved symbol: %s", self.resolved_symbol)
            return True
        except ImportError:
            log.error("MetaTrader5 package not installed")
            return False
        except Exception as exc:
            log.error("MT5 connection error: %s", exc)
            return False

    def connect_with_retry(self) -> bool:
        delay = self.reconnect_backoff_seconds
        for attempt in range(1, self.reconnect_attempts + 1):
            log.info("MT5 connect attempt %d/%d...", attempt, self.reconnect_attempts)
            if self.connect():
                return True
            if attempt < self.reconnect_attempts:
                log.warning("Retrying in %.1fs...", delay)
                time.sleep(delay)
                delay *= self.reconnect_backoff_multiplier
        log.error("MT5 connection failed after %d attempts", self.reconnect_attempts)
        return False

    def resolve_symbol(self) -> Optional[str]:
        if not self._mt5:
            return None
        for candidate in self.symbol_candidates:
            info = self._mt5.symbol_info(candidate)
            if info is not None:
                if not info.visible:
                    self._mt5.symbol_select(candidate, True)
                return candidate
        return None

    def ensure_connected(self) -> bool:
        if self._connected and self._mt5 is not None and self._mt5.terminal_info() is not None:
            return True
        log.warning("MT5 connection down — reconnecting")
        self._connected = False
        return self.connect_with_retry()

    def health_check(self) -> Dict[str, Any]:
        terminal_ok = False
        account_ok = False
        if self._mt5 is not None:
            try:
                terminal_ok = self._mt5.terminal_info() is not None
                account_ok = self._mt5.account_info() is not None
            except Exception as exc:
                log.error("Health check error: %s", exc)
        status = {
            "connected": self._connected and terminal_ok,
            "symbol": self.resolved_symbol,
            "terminal_ok": terminal_ok,
            "account_ok": account_ok,
            "checked_at": datetime.now(),
        }
        self._last_health_check = status
        if not status["connected"]:
            log.warning("Health check failed: %s", status)
        return status

    def disconnect(self) -> bool:
        if self._mt5:
            self._mt5.shutdown()
        self._connected = False
        return True

    def is_connected(self) -> bool:
        return self._connected

    def get_account_info(self) -> AccountInfo:
        if not self.ensure_connected():
            return AccountInfo(0, 0, 0, 0, 0)
        try:
            info = self._mt5.account_info()
        except Exception as exc:
            log.error("get_account_info exception: %s", exc)
            return AccountInfo(0, 0, 0, 0, 0)
        if info is None:
            log.error("get_account_info: no data")
            return AccountInfo(0, 0, 0, 0, 0)
        return AccountInfo(
            balance=info.balance,
            equity=info.equity,
            margin=info.margin,
            free_margin=info.margin_free,
            margin_level=info.margin_level,
            currency=info.currency,
            leverage=self.leverage,
        )

    def _resolve_timeframe(self, timeframe: str):
        tf_map = {
            "M1": self._mt5.TIMEFRAME_M1, "M5": self._mt5.TIMEFRAME_M5,
            "M15": self._mt5.TIMEFRAME_M15, "M30": self._mt5.TIMEFRAME_M30,
            "H1": self._mt5.TIMEFRAME_H1, "H4": self._mt5.TIMEFRAME_H4,
            "D1": self._mt5.TIMEFRAME_D1,
        }
        if timeframe not in tf_map:
            log.warning("Unknown timeframe %s, falling back to H1", timeframe)
        return tf_map.get(timeframe, self._mt5.TIMEFRAME_H1)

    @staticmethod
    def _rates_to_dataframe(rates) -> pd.DataFrame:
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        return df.rename(columns={
            "time": "Date", "open": "Open", "high": "High",
            "low": "Low", "close": "Close", "tick_volume": "Volume",
        })

    def get_ohlcv(self, symbol: Optional[str] = None, timeframe: str = "H1", bars: int = 500) -> pd.DataFrame:
        if bars <= 0:
            raise ValueError("bars must be > 0")
        if not self.ensure_connected():
            return pd.DataFrame()
        symbol = symbol or self.resolved_symbol
        if not symbol:
            log.error("No symbol available")
            return pd.DataFrame()
        tf = self._resolve_timeframe(timeframe)
        try:
            rates = self._mt5.copy_rates_from_pos(symbol, tf, 0, bars)
        except Exception as exc:
            log.error("get_ohlcv exception: %s", exc)
            return pd.DataFrame()
        if rates is None or len(rates) == 0:
            log.error("No OHLCV data for %s %s", symbol, timeframe)
            return pd.DataFrame()
        return self._rates_to_dataframe(rates)

    def load_historical_data(self, start: datetime, end: datetime, symbol: Optional[str] = None, timeframe: str = "H1") -> pd.DataFrame:
        if start >= end:
            raise ValueError("start must be before end")
        if not self.ensure_connected():
            return pd.DataFrame()
        symbol = symbol or self.resolved_symbol
        if not symbol:
            return pd.DataFrame()
        tf = self._resolve_timeframe(timeframe)
        try:
            rates = self._mt5.copy_rates_range(symbol, tf, start, end)
        except Exception as exc:
            log.error("load_historical_data exception: %s", exc)
            return pd.DataFrame()
        if not rates:
            return pd.DataFrame()
        return self._rates_to_dataframe(rates)

    def get_price(self, symbol: Optional[str] = None) -> Dict[str, float]:
        if not self.ensure_connected():
            return {"bid": 0.0, "ask": 0.0}
        symbol = symbol or self.resolved_symbol
        try:
            tick = self._mt5.symbol_info_tick(symbol)
        except Exception as exc:
            log.error("get_price exception: %s", exc)
            return {"bid": 0.0, "ask": 0.0}
        if tick:
            return {"bid": tick.bid, "ask": tick.ask, "time": tick.time}
        return {"bid": 0.0, "ask": 0.0}

    def send_order(self, signal: TradeSignal) -> Dict[str, Any]:
        if not self.ensure_connected():
            return {"success": False, "error": "MT5 not connected"}
        try:
            sym_info = self._mt5.symbol_info(signal.symbol)
            if sym_info is None:
                return {"success": False, "error": f"Symbol {signal.symbol} not found"}
            if not sym_info.visible:
                if not self._mt5.symbol_select(signal.symbol, True):
                    return {"success": False, "error": f"Failed to select {signal.symbol}"}
            tick = self._mt5.symbol_info_tick(signal.symbol)
            if tick is None:
                return {"success": False, "error": f"No tick data for {signal.symbol}"}
            order_type = self._mt5.ORDER_TYPE_BUY if signal.direction == "BUY" else self._mt5.ORDER_TYPE_SELL
            price = tick.ask if signal.direction == "BUY" else tick.bid
            request = {
                "action": self._mt5.TRADE_ACTION_DEAL,
                "symbol": signal.symbol,
                "volume": signal.lot_size,
                "type": order_type,
                "price": price,
                "sl": signal.sl,
                "tp": signal.tp,
                "deviation": 10,
                "magic": self.magic_number,
                "comment": f"GoldBot_{signal.strategy}",
                "type_time": self._mt5.ORDER_TIME_GTC,
                "type_filling": self._mt5.ORDER_FILLING_IOC,
            }
            result = self._mt5.order_send(request)
            if result is None:
                err = self._mt5.last_error()
                log.error("send_order: no result, error=%s", err)
                return {"success": False, "error": f"order_send failed: {err}"}
            if result.retcode == self._mt5.TRADE_RETCODE_DONE:
                log.info("Order executed: %s %s %.2f lots @ %.2f (ticket=%d)",
                         signal.direction, signal.symbol, signal.lot_size, result.price, result.order)
                return {"success": True, "ticket": result.order, "fill_price": result.price}
            log.error("send_order failed: retcode=%s", result.retcode)
            return {"success": False, "error": f"Order failed: {result.retcode}"}
        except Exception as exc:
            log.error("send_order exception: %s", exc)
            return {"success": False, "error": f"Exception: {exc}"}

    def get_positions(self) -> List[Position]:
        if not self.ensure_connected():
            return []
        try:
            positions = self._mt5.positions_get()
            if positions is None:
                return []
            result = []
            for p in positions:
                pos = Position(
                    ticket=p.ticket, symbol=p.symbol,
                    position_type="BUY" if p.type == 0 else "SELL",
                    volume=p.volume, open_price=p.price_open,
                    sl=p.sl, tp=p.tp, profit=p.profit, swap=p.swap,
                    open_time=datetime.fromtimestamp(p.time),
                    magic_number=getattr(p, "magic", 0),
                )
                result.append(pos)
                self._position_cache[pos.ticket] = pos
            return result
        except Exception as exc:
            log.error("get_positions exception: %s", exc)
            return []

    def _sync_trade_history(self) -> None:
        if not self.ensure_connected():
            return
        since = self._last_deal_sync or (datetime.now() - timedelta(days=1))
        now = datetime.now()
        try:
            deals = self._mt5.history_deals_get(since, now)
        except Exception as exc:
            log.error("_sync_trade_history exception: %s", exc)
            return
        self._last_deal_sync = now
        if not deals:
            return
        out_entry = getattr(self._mt5, "DEAL_ENTRY_OUT", 1)
        for d in deals:
            if getattr(d, "magic", None) != self.magic_number:
                continue
            if getattr(d, "entry", None) != out_entry:
                continue
            position_id = getattr(d, "position_id", None)
            if position_id is None or position_id in self._closed_position_ids:
                continue
            snapshot = self._position_cache.pop(position_id, None)
            record = {
                "ticket": position_id, "type": "CLOSE",
                "direction": snapshot.position_type if snapshot else ("SELL" if d.type == 0 else "BUY"),
                "symbol": d.symbol, "open_price": snapshot.open_price if snapshot else None,
                "close_price": d.price, "sl": snapshot.sl if snapshot else None,
                "tp": snapshot.tp if snapshot else None, "lot_size": d.volume,
                "pl": d.profit, "open_time": snapshot.open_time if snapshot else None,
                "time": datetime.fromtimestamp(d.time),
            }
            self._trade_history.append(record)
            self._closed_position_ids.add(position_id)
            log.info("Trade history sync: position #%s closed P/L=$%.2f", position_id, d.profit)

    def get_trade_history(self) -> List[Dict[str, Any]]:
        self._sync_trade_history()
        return list(self._trade_history)

    def close_position(self, ticket: int) -> bool:
        if not self.ensure_connected():
            return False
        try:
            position = self._mt5.positions_get(ticket=ticket)
            if not position:
                log.warning("close_position: ticket #%d not found", ticket)
                return False
            pos = position[0]
            tick = self._mt5.symbol_info_tick(pos.symbol)
            if tick is None:
                return False
            price = tick.bid if pos.type == 0 else tick.ask
            order_type = self._mt5.ORDER_TYPE_SELL if pos.type == 0 else self._mt5.ORDER_TYPE_BUY
            request = {
                "action": self._mt5.TRADE_ACTION_DEAL,
                "symbol": pos.symbol, "volume": pos.volume, "type": order_type,
                "position": pos.ticket, "price": price, "deviation": 10,
                "magic": self.magic_number, "comment": "GoldBot_Close",
                "type_time": self._mt5.ORDER_TIME_GTC,
                "type_filling": self._mt5.ORDER_FILLING_IOC,
            }
            result = self._mt5.order_send(request)
            if result is None:
                return False
            ok = result.retcode == self._mt5.TRADE_RETCODE_DONE
            if ok:
                log.info("Position #%d closed at %.2f", ticket, price)
            return ok
        except Exception as exc:
            log.error("close_position exception: %s", exc)
            return False

    def close_all_positions(self) -> bool:
        all_ok = True
        for pos in self.get_positions():
            if not self.close_position(pos.ticket):
                all_ok = False
        return all_ok

    def modify_position_sl_tp(self, ticket: int, sl: Optional[float] = None, tp: Optional[float] = None) -> bool:
        if not self.ensure_connected():
            return False
        position = self._mt5.positions_get(ticket=ticket)
        if not position:
            return False
        pos = position[0]
        request = {
            "action": self._mt5.TRADE_ACTION_SLTP,
            "symbol": pos.symbol,
            "position": ticket,
            "sl": sl if sl is not None else pos.sl,
            "tp": tp if tp is not None else pos.tp,
        }
        result = self._mt5.order_send(request)
        ok = result is not None and result.retcode == self._mt5.TRADE_RETCODE_DONE
        if ok:
            log.info("MT5 position #%d modified SL=%s TP=%s", ticket, sl, tp)
        return ok

    def close_position_partial(self, ticket: int, volume: float) -> Dict[str, Any]:
        if not self.ensure_connected():
            return {"success": False, "error": "MT5 not connected"}
        position = self._mt5.positions_get(ticket=ticket)
        if not position:
            return {"success": False, "error": "Position not found"}
        pos = position[0]
        close_volume = min(volume, pos.volume)
        tick = self._mt5.symbol_info_tick(pos.symbol)
        if tick is None:
            return {"success": False, "error": "No tick data"}
        price = tick.bid if pos.type == 0 else tick.ask
        order_type = self._mt5.ORDER_TYPE_SELL if pos.type == 0 else self._mt5.ORDER_TYPE_BUY
        request = {
            "action": self._mt5.TRADE_ACTION_DEAL,
            "symbol": pos.symbol, "volume": close_volume, "type": order_type,
            "position": pos.ticket, "price": price, "deviation": 10,
            "magic": self.magic_number, "comment": "GoldBot_PartialClose",
            "type_time": self._mt5.ORDER_TIME_GTC,
            "type_filling": self._mt5.ORDER_FILLING_IOC,
        }
        result = self._mt5.order_send(request)
        ok = result is not None and result.retcode == self._mt5.TRADE_RETCODE_DONE
        if ok:
            return {"success": True, "closed_volume": close_volume, "remaining_volume": pos.volume - close_volume}
        return {"success": False, "error": f"Partial close failed: {getattr(result, 'retcode', 'no result')}"}


class MT5BridgeBroker(BrokerConnector):
    """MT5 access over HTTP, via a small Windows-side bridge service.

    Use this instead of MT5Broker when the app itself runs on Linux/Docker
    (the `MetaTrader5` package only works on Windows next to a real
    terminal — it cannot be imported here). This class makes plain HTTP
    calls to that bridge and implements the same BrokerConnector interface
    as MT5Broker, so it's a drop-in replacement everywhere a broker is used.

    See mt5_bridge/ (separate service, runs on Windows) for the server side.
    """

    def __init__(
        self,
        base_url: str,
        token: str = "",
        symbol_candidates: Tuple[str, ...] = ("XAUUSD", "XAUUSDm", "XAUUSD.a", "XAUUSD.raw", "GOLD"),
        reconnect_attempts: int = 5,
        reconnect_backoff_seconds: float = 5.0,
        reconnect_backoff_multiplier: float = 2.0,
        request_timeout_seconds: float = 10.0,
        leverage: float = 100.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.symbol_candidates = symbol_candidates
        self.reconnect_attempts = reconnect_attempts
        self.reconnect_backoff_seconds = reconnect_backoff_seconds
        self.reconnect_backoff_multiplier = reconnect_backoff_multiplier
        self.request_timeout_seconds = request_timeout_seconds
        self.leverage = leverage

        self._connected = False
        self.resolved_symbol: Optional[str] = None
        self._last_health_check: Optional[Dict[str, Any]] = None

    # -- low-level HTTP helper --------------------------------------------

    def _headers(self) -> Dict[str, str]:
        return {"X-Bridge-Token": self.token} if self.token else {}

    def _request(self, method: str, path: str, **kwargs) -> Any:
        import requests  # local import: keeps requests optional for non-bridge deployments
        url = f"{self.base_url}{path}"
        try:
            resp = requests.request(
                method, url, headers=self._headers(),
                timeout=self.request_timeout_seconds, **kwargs,
            )
        except requests.exceptions.RequestException as exc:
            raise ConnectionError(f"Bridge unreachable at {url}: {exc}") from exc
        if resp.status_code >= 400:
            try:
                detail = resp.json().get("detail", resp.text)
            except Exception:
                detail = resp.text
            raise BrokerError(f"Bridge error {resp.status_code} on {path}: {detail}")
        return resp.json()

    # -- BrokerConnector interface ------------------------------------------

    def connect(self) -> bool:
        try:
            data = self._request("POST", "/connect", json={})
        except (ConnectionError, BrokerError) as exc:
            log.error("MT5BridgeBroker connect failed: %s", exc)
            self._connected = False
            return False
        self._connected = bool(data.get("connected"))
        if self._connected:
            log.info("MT5BridgeBroker connected via %s (balance=%s)", self.base_url, data.get("balance"))
            self.resolved_symbol = self.resolve_symbol()
        return self._connected

    def connect_with_retry(self) -> bool:
        delay = self.reconnect_backoff_seconds
        for attempt in range(1, self.reconnect_attempts + 1):
            log.info("Bridge connect attempt %d/%d...", attempt, self.reconnect_attempts)
            if self.connect():
                return True
            if attempt < self.reconnect_attempts:
                log.warning("Retrying bridge connection in %.1fs...", delay)
                time.sleep(delay)
                delay *= self.reconnect_backoff_multiplier
        log.error("Bridge connection failed after %d attempts", self.reconnect_attempts)
        return False

    def resolve_symbol(self) -> Optional[str]:
        # The bridge doesn't expose a symbol-lookup endpoint yet, so trust
        # the first candidate; get_ohlcv/get_price will surface a clear
        # error if it's wrong for this broker's naming convention.
        return self.symbol_candidates[0] if self.symbol_candidates else None

    def ensure_connected(self) -> bool:
        if self._connected:
            try:
                status = self._request("GET", "/status")
                if status.get("connected"):
                    return True
            except (ConnectionError, BrokerError):
                pass
        log.warning("Bridge connection down — reconnecting")
        self._connected = False
        return self.connect_with_retry()

    def health_check(self) -> Dict[str, Any]:
        try:
            status = self._request("GET", "/status")
        except (ConnectionError, BrokerError) as exc:
            status = {"connected": False, "terminal_ok": False, "account_ok": False, "last_error": str(exc)}
        result = {
            "connected": bool(status.get("connected")),
            "symbol": self.resolved_symbol,
            "terminal_ok": bool(status.get("terminal_ok")),
            "account_ok": bool(status.get("account_ok")),
            "checked_at": datetime.now(),
        }
        self._last_health_check = result
        if not result["connected"]:
            log.warning("Bridge health check failed: %s", status)
        return result

    def disconnect(self) -> bool:
        try:
            self._request("POST", "/disconnect")
        except (ConnectionError, BrokerError) as exc:
            log.warning("Bridge disconnect call failed (continuing): %s", exc)
        self._connected = False
        return True

    def is_connected(self) -> bool:
        return self._connected

    def get_account_info(self) -> AccountInfo:
        if not self.ensure_connected():
            return AccountInfo(0, 0, 0, 0, 0)
        status = self._request("GET", "/status")
        account = status.get("account") or {}
        if not account:
            log.error("get_account_info: bridge returned no account data")
            return AccountInfo(0, 0, 0, 0, 0)
        return AccountInfo(
            balance=account.get("balance", 0.0),
            equity=account.get("equity", 0.0),
            margin=account.get("margin", 0.0),
            free_margin=account.get("free_margin", 0.0),
            margin_level=account.get("margin_level", 0.0),
            currency=account.get("currency", "USD"),
            leverage=self.leverage,
        )

    def get_price(self, symbol: Optional[str] = None) -> Dict[str, float]:
        symbol = symbol or self.resolved_symbol
        if not symbol:
            return {"bid": 0.0, "ask": 0.0}
        # Cheapest way to get a live price from the bridge today is the
        # latest OHLCV candle's close; good enough for a spread/price widget.
        try:
            df = self.get_ohlcv(symbol=symbol, timeframe="M1", bars=1)
        except Exception as exc:
            log.error("get_price exception: %s", exc)
            return {"bid": 0.0, "ask": 0.0}
        if df.empty:
            return {"bid": 0.0, "ask": 0.0}
        close = float(df.iloc[-1]["Close"])
        return {"bid": close, "ask": close}

    def get_ohlcv(self, symbol: Optional[str] = None, timeframe: str = "H1", bars: int = 500) -> pd.DataFrame:
        if bars <= 0:
            raise ValueError("bars must be > 0")
        if not self.ensure_connected():
            return pd.DataFrame()
        symbol = symbol or self.resolved_symbol
        if not symbol:
            log.error("No symbol available")
            return pd.DataFrame()
        try:
            data = self._request("GET", "/ohlcv", params={"symbol": symbol, "timeframe": timeframe, "bars": bars})
        except (ConnectionError, BrokerError) as exc:
            log.error("get_ohlcv exception: %s", exc)
            return pd.DataFrame()
        return self._candles_to_dataframe(data.get("candles", []))

    def load_historical_data(self, start: datetime, end: datetime, symbol: Optional[str] = None, timeframe: str = "H1") -> pd.DataFrame:
        if start >= end:
            raise ValueError("start must be before end")
        if not self.ensure_connected():
            return pd.DataFrame()
        symbol = symbol or self.resolved_symbol
        if not symbol:
            return pd.DataFrame()
        try:
            data = self._request("GET", "/ohlcv/range", params={
                "symbol": symbol, "timeframe": timeframe,
                "start": start.isoformat(), "end": end.isoformat(),
            })
        except (ConnectionError, BrokerError) as exc:
            log.error("load_historical_data exception: %s", exc)
            return pd.DataFrame()
        return self._candles_to_dataframe(data.get("candles", []))

    @staticmethod
    def _candles_to_dataframe(candles: List[Dict[str, Any]]) -> pd.DataFrame:
        if not candles:
            return pd.DataFrame()
        df = pd.DataFrame(candles)
        df["Date"] = pd.to_datetime(df["time"])
        return df.rename(columns={
            "open": "Open", "high": "High", "low": "Low",
            "close": "Close", "volume": "Volume",
        })[["Date", "Open", "High", "Low", "Close", "Volume"]]

    def send_order(self, signal: TradeSignal) -> Dict[str, Any]:
        if not self.ensure_connected():
            return {"success": False, "error": "Bridge not connected"}
        try:
            return self._request("POST", "/order", json={
                "symbol": signal.symbol, "direction": signal.direction,
                "volume": signal.lot_size, "sl": signal.sl, "tp": signal.tp,
                "comment": f"GoldBot_{signal.strategy}",
            })
        except (ConnectionError, BrokerError) as exc:
            log.error("send_order exception: %s", exc)
            return {"success": False, "error": str(exc)}

    def get_positions(self) -> List[Position]:
        if not self.ensure_connected():
            return []
        try:
            data = self._request("GET", "/positions")
        except (ConnectionError, BrokerError) as exc:
            log.error("get_positions exception: %s", exc)
            return []
        return [
            Position(
                ticket=p["ticket"], symbol=p["symbol"], position_type=p["direction"],
                volume=p["volume"], open_price=p["open_price"], sl=p["sl"], tp=p["tp"],
                profit=p["profit"], swap=p["swap"],
                open_time=datetime.fromisoformat(p["open_time"]),
                magic_number=p.get("magic", 0),
            )
            for p in data
        ]

    def close_position(self, ticket: int) -> bool:
        if not self.ensure_connected():
            return False
        try:
            result = self._request("POST", "/order/close", json={"ticket": ticket})
        except (ConnectionError, BrokerError) as exc:
            log.error("close_position exception: %s", exc)
            return False
        return bool(result.get("success"))

    def close_all_positions(self) -> bool:
        all_ok = True
        for pos in self.get_positions():
            if not self.close_position(pos.ticket):
                all_ok = False
        return all_ok

    def modify_position_sl_tp(self, ticket: int, sl: Optional[float] = None, tp: Optional[float] = None) -> bool:
        if not self.ensure_connected():
            return False
        try:
            result = self._request("POST", "/order/modify", json={"ticket": ticket, "sl": sl, "tp": tp})
        except (ConnectionError, BrokerError) as exc:
            log.error("modify_position_sl_tp exception: %s", exc)
            return False
        return bool(result.get("success"))

    def close_position_partial(self, ticket: int, volume: float) -> Dict[str, Any]:
        if not self.ensure_connected():
            return {"success": False, "error": "Bridge not connected"}
        try:
            return self._request("POST", "/order/close", json={"ticket": ticket, "volume": volume})
        except (ConnectionError, BrokerError) as exc:
            log.error("close_position_partial exception: %s", exc)
            return {"success": False, "error": str(exc)}
