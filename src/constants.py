"""Application constants and enums."""

from enum import Enum


class Action(str, Enum):
    """Trade actions."""
    BUY = "BUY"
    SELL = "SELL"
    NO_TRADE = "NO_TRADE"


class Regime(str, Enum):
    """Market regime classifications."""
    STRONG_UPTREND = "strong_uptrend"
    WEAK_UPTREND = "weak_uptrend"
    STRONG_DOWNTREND = "strong_downtrend"
    WEAK_DOWNTREND = "weak_downtrend"
    RANGING = "ranging"
    HIGH_VOLATILITY_BREAKOUT = "high_volatility_breakout"
    LOW_VOLATILITY = "low_volatility"
    UNCLEAR = "unclear"


class BrokerType(str, Enum):
    """Supported broker types."""
    PAPER = "paper"
    MT5 = "mt5"


class NotificationChannel(str, Enum):
    """Notification channels."""
    TELEGRAM = "telegram"
    DISCORD = "discord"
