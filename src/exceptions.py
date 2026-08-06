"""Custom exceptions for the trading bot."""


class TradingBotError(Exception):
    """Base exception for all trading bot errors."""
    pass


class ConfigError(TradingBotError):
    """Raised when configuration is invalid or missing."""
    pass


class BrokerError(TradingBotError):
    """Raised when broker operations fail."""
    pass


class ConnectionError(BrokerError):
    """Raised when broker connection fails."""
    pass


class OrderError(BrokerError):
    """Raised when order execution fails."""
    pass


class RiskError(TradingBotError):
    """Raised when risk limits are breached."""
    pass


class DataError(TradingBotError):
    """Raised when data fetching or processing fails."""
    pass
