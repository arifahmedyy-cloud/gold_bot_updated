"""Input validation utilities.

Prevents invalid/malicious inputs from reaching trading logic.
"""

from __future__ import annotations

from typing import Optional, Tuple
import re

from src.logger import get_logger

log = get_logger(__name__)


class ValidationError(ValueError):
    """Raised when input validation fails."""
    pass


def validate_symbol(symbol: str) -> str:
    """Validate and normalize trading symbol.

    Args:
        symbol: Raw symbol string.

    Returns:
        Normalized symbol.

    Raises:
        ValidationError: If symbol is invalid.
    """
    if not symbol or not isinstance(symbol, str):
        raise ValidationError("Symbol must be a non-empty string")
    symbol = symbol.strip().upper()
    if not re.match(r"^[A-Z0-9]{3,20}(\.[A-Z]+)?$", symbol):
        raise ValidationError(f"Invalid symbol format: {symbol}")
    return symbol


def validate_price(price: float, name: str = "price") -> float:
    """Validate price value.

    Args:
        price: Price value.
        name: Field name for error messages.

    Returns:
        Validated price.

    Raises:
        ValidationError: If price is invalid.
    """
    if not isinstance(price, (int, float)):
        raise ValidationError(f"{name} must be numeric")
    if price <= 0:
        raise ValidationError(f"{name} must be positive, got {price}")
    if price > 1_000_000:
        raise ValidationError(f"{name} seems unrealistic: {price}")
    return float(price)


def validate_lot_size(lot: float, max_lot: float = 100.0) -> float:
    """Validate lot size.

    Args:
        lot: Lot size value.
        max_lot: Maximum allowed lot size.

    Returns:
        Validated lot size.

    Raises:
        ValidationError: If lot size is invalid.
    """
    if not isinstance(lot, (int, float)):
        raise ValidationError("Lot size must be numeric")
    if lot <= 0:
        raise ValidationError("Lot size must be positive")
    if lot > max_lot:
        raise ValidationError(f"Lot size {lot} exceeds maximum {max_lot}")
    return round(float(lot), 2)


def validate_risk_pct(risk: float) -> float:
    """Validate risk percentage.

    Args:
        risk: Risk percentage.

    Returns:
        Validated risk percentage.

    Raises:
        ValidationError: If risk is invalid.
    """
    if not isinstance(risk, (int, float)):
        raise ValidationError("Risk must be numeric")
    if risk <= 0 or risk > 100:
        raise ValidationError(f"Risk must be between 0 and 100, got {risk}")
    return float(risk)


def validate_sl_tp(entry: float, sl: float, tp: float, direction: str) -> Tuple[float, float, float]:
    """Validate stop loss and take profit relative to entry.

    Args:
        entry: Entry price.
        sl: Stop loss price.
        tp: Take profit price.
        direction: Trade direction (BUY or SELL).

    Returns:
        Tuple of (entry, sl, tp).

    Raises:
        ValidationError: If SL/TP logic is invalid.
    """
    entry = validate_price(entry, "entry")
    sl = validate_price(sl, "stop loss")
    tp = validate_price(tp, "take profit")

    if direction == "BUY":
        if sl >= entry:
            raise ValidationError(f"BUY SL ({sl}) must be below entry ({entry})")
        if tp <= entry:
            raise ValidationError(f"BUY TP ({tp}) must be above entry ({entry})")
    elif direction == "SELL":
        if sl <= entry:
            raise ValidationError(f"SELL SL ({sl}) must be above entry ({entry})")
        if tp >= entry:
            raise ValidationError(f"SELL TP ({tp}) must be below entry ({entry})")
    else:
        raise ValidationError(f"Invalid direction: {direction}")

    risk = abs(entry - sl)
    reward = abs(tp - entry)
    if reward < risk:
        log.warning("Reward (%s) < Risk (%s) — poor R:R ratio", reward, risk)

    return entry, sl, tp


def validate_mt5_credentials(login: int, password: str, server: str) -> Tuple[int, str, str]:
    """Validate MT5 credentials.

    Args:
        login: Account login number.
        password: Account password.
        server: Broker server name.

    Returns:
        Tuple of (login, password, server).

    Raises:
        ValidationError: If credentials are invalid.
    """
    if not isinstance(login, int) or login <= 0:
        raise ValidationError("MT5 login must be a positive integer")
    if not password or len(password) < 4:
        raise ValidationError("MT5 password too short")
    if not server or len(server) < 3:
        raise ValidationError("MT5 server name required")
    return login, password, server


def sanitize_api_key(key: str) -> str:
    """Sanitize and validate API key format.

    Args:
        key: Raw API key.

    Returns:
        Cleaned key or empty string.
    """
    if not key or not isinstance(key, str):
        return ""
    key = key.strip()
    if len(key) < 10:
        return ""
    return key
