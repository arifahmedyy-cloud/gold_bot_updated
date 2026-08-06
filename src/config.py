"""Centralized configuration management.

Loads settings from environment variables and .env files with validation.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, Tuple

from src.exceptions import ConfigError


def _env_bool(name: str, default: bool = False) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def _env_float(name: str, default: float) -> float:
    val = os.environ.get(name)
    return float(val) if val not in (None, "") else default


def _env_int(name: str, default: int) -> int:
    val = os.environ.get(name)
    return int(val) if val not in (None, "") else default


def _env_str(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


@dataclass
class MT5Config:
    """MetaTrader 5 configuration."""
    login: int = 0
    password: str = ""
    server: str = ""
    symbol_candidates: Tuple[str, ...] = (
        "XAUUSD", "XAUUSDm", "XAUUSD.a", "XAUUSD.raw", "GOLD"
    )
    timeframe: str = "H1"
    bars: int = 500
    reconnect_attempts: int = 5
    reconnect_backoff_seconds: float = 5.0
    reconnect_backoff_multiplier: float = 2.0
    health_check_interval_seconds: int = 30
    leverage: float = 100.0


@dataclass
class MT5BridgeConfig:
    """Config for talking to the Windows-side MT5 bridge service over HTTP,
    used when broker='mt5_bridge' (i.e. the app itself runs on Linux/Docker)."""
    base_url: str = ""
    token: str = ""
    symbol_candidates: Tuple[str, ...] = (
        "XAUUSD", "XAUUSDm", "XAUUSD.a", "XAUUSD.raw", "GOLD"
    )
    reconnect_attempts: int = 5
    reconnect_backoff_seconds: float = 5.0
    reconnect_backoff_multiplier: float = 2.0
    request_timeout_seconds: float = 10.0
    leverage: float = 100.0


@dataclass
class RiskConfig:
    """Risk management configuration."""
    base_risk_pct: float = 1.0
    max_risk_pct: float = 2.0
    max_consecutive_losses: int = 3
    drawdown_reduction_factor: float = 0.5
    daily_loss_limit_pct: float = 5.0
    daily_profit_lock_pct: Optional[float] = None
    breakeven_trigger_r: float = 1.0
    breakeven_buffer_pct: float = 0.05
    partial_close_trigger_r: float = 1.0
    partial_close_fraction: float = 0.5
    trailing_trigger_r: float = 1.5
    trailing_distance_r: float = 0.5
    use_auto_drawdown_risk: bool = True
    use_kelly_sizing: bool = False
    use_trailing_stop: bool = True
    max_net_usd_exposure: float = 2.0
    kelly_min_trades: int = 10
    kelly_fraction: float = 0.5
    min_ai_score: int = 55
    min_ml_confidence: float = 60.0
    enable_ml_filter: bool = True
    max_open_trades: int = 1


@dataclass
class NotificationConfig:
    """Alert notification configuration."""
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    discord_webhook_url: str = ""
    enabled: bool = False


@dataclass
class NewsConfig:
    """News and sentiment configuration."""
    alpha_vantage_api_key: str = ""
    enabled: bool = False


@dataclass
class AIConfig:
    """AI assistant configuration."""
    provider: str = "claude"          # "claude" | "gemini" | "gpt"
    anthropic_api_key: str = ""
    google_api_key: str = ""
    openai_api_key: str = ""
    model: str = "claude-sonnet-5"
    gemini_model: str = "gemini-2.0-flash"
    gpt_model: str = "gpt-4o-mini"
    enabled: bool = False


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    log_dir: str = "logs"
    log_filename: str = "trading.log"
    max_bytes: int = 5 * 1024 * 1024
    backup_count: int = 5
    console: bool = True


@dataclass
class BotConfig:
    """Root configuration container."""
    broker: str = "paper"
    mt5: MT5Config = field(default_factory=MT5Config)
    mt5_bridge: MT5BridgeConfig = field(default_factory=MT5BridgeConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    news: NewsConfig = field(default_factory=NewsConfig)
    ai: AIConfig = field(default_factory=AIConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    notifications: NotificationConfig = field(default_factory=NotificationConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_safe_dict(self) -> Dict[str, Any]:
        """Config dict with secrets masked — safe to display in the UI."""
        d = asdict(self)
        if d["mt5"].get("password"):
            d["mt5"]["password"] = "••••••••"
        if d["mt5_bridge"].get("token"):
            d["mt5_bridge"]["token"] = "••••" + d["mt5_bridge"]["token"][-4:]
        if d["ai"].get("anthropic_api_key"):
            d["ai"]["anthropic_api_key"] = "••••" + d["ai"]["anthropic_api_key"][-4:]
        if d["ai"].get("google_api_key"):
            d["ai"]["google_api_key"] = "••••" + d["ai"]["google_api_key"][-4:]
        if d["ai"].get("openai_api_key"):
            d["ai"]["openai_api_key"] = "••••" + d["ai"]["openai_api_key"][-4:]
        if d["news"].get("alpha_vantage_api_key"):
            d["news"]["alpha_vantage_api_key"] = "••••" + d["news"]["alpha_vantage_api_key"][-4:]
        if d["notifications"].get("telegram_bot_token"):
            d["notifications"]["telegram_bot_token"] = "••••" + d["notifications"]["telegram_bot_token"][-4:]
        if d["notifications"].get("discord_webhook_url"):
            d["notifications"]["discord_webhook_url"] = "••••webhook-hidden••••"
        return d

    def validate(self) -> None:
        if self.broker not in ("paper", "mt5", "mt5_bridge"):
            raise ConfigError(f"Unknown broker '{self.broker}', expected 'paper', 'mt5', or 'mt5_bridge'")
        if self.broker == "mt5":
            if not self.mt5.server:
                raise ConfigError("MT5 server is required (set MT5_SERVER)")
            if self.mt5.login == 0:
                raise ConfigError("MT5 login is required (set MT5_LOGIN)")
            if not self.mt5.password:
                raise ConfigError("MT5 password is required (set MT5_PASSWORD)")
        if self.broker == "mt5_bridge":
            if not self.mt5_bridge.base_url:
                raise ConfigError("Bridge URL is required (set MT5_BRIDGE_URL), e.g. http://192.168.1.50:8800")
        if not (0 < self.risk.base_risk_pct <= self.risk.max_risk_pct):
            raise ConfigError("risk.base_risk_pct must be > 0 and <= risk.max_risk_pct")
        if self.news.enabled and not self.news.alpha_vantage_api_key:
            raise ConfigError("News filter enabled but ALPHA_VANTAGE_API_KEY is missing")
        if self.ai.enabled and not self.ai.anthropic_api_key:
            raise ConfigError("AI assistant enabled but ANTHROPIC_API_KEY is missing")


def _deep_merge(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = _deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged


def _config_from_env() -> BotConfig:
    mt5 = MT5Config(
        login=_env_int("MT5_LOGIN", 0),
        password=_env_str("MT5_PASSWORD"),
        server=_env_str("MT5_SERVER"),
        timeframe=_env_str("MT5_TIMEFRAME", "H1"),
        bars=_env_int("MT5_BARS", 500),
        reconnect_attempts=_env_int("MT5_RECONNECT_ATTEMPTS", 5),
        reconnect_backoff_seconds=_env_float("MT5_RECONNECT_BACKOFF_SECONDS", 5.0),
        reconnect_backoff_multiplier=_env_float("MT5_RECONNECT_BACKOFF_MULTIPLIER", 2.0),
        health_check_interval_seconds=_env_int("MT5_HEALTH_CHECK_INTERVAL", 30),
        leverage=_env_float("MT5_LEVERAGE", 100.0),
    )
    if os.environ.get("MT5_SYMBOL_CANDIDATES"):
        mt5.symbol_candidates = tuple(
            s.strip() for s in os.environ["MT5_SYMBOL_CANDIDATES"].split(",") if s.strip()
        )

    mt5_bridge = MT5BridgeConfig(
        base_url=_env_str("MT5_BRIDGE_URL", ""),
        token=_env_str("MT5_BRIDGE_TOKEN", ""),
        reconnect_attempts=_env_int("MT5_BRIDGE_RECONNECT_ATTEMPTS", 5),
        reconnect_backoff_seconds=_env_float("MT5_BRIDGE_RECONNECT_BACKOFF_SECONDS", 5.0),
        reconnect_backoff_multiplier=_env_float("MT5_BRIDGE_RECONNECT_BACKOFF_MULTIPLIER", 2.0),
        request_timeout_seconds=_env_float("MT5_BRIDGE_REQUEST_TIMEOUT_SECONDS", 10.0),
        leverage=_env_float("MT5_BRIDGE_LEVERAGE", 100.0),
    )
    if os.environ.get("MT5_SYMBOL_CANDIDATES"):
        mt5_bridge.symbol_candidates = tuple(
            s.strip() for s in os.environ["MT5_SYMBOL_CANDIDATES"].split(",") if s.strip()
        )

    risk = RiskConfig(
        base_risk_pct=_env_float("RISK_BASE_PCT", 1.0),
        max_risk_pct=_env_float("RISK_MAX_PCT", 2.0),
        max_consecutive_losses=_env_int("RISK_MAX_CONSECUTIVE_LOSSES", 3),
        drawdown_reduction_factor=_env_float("RISK_DRAWDOWN_REDUCTION_FACTOR", 0.5),
        daily_loss_limit_pct=_env_float("RISK_DAILY_LOSS_LIMIT_PCT", 5.0),
        daily_profit_lock_pct=(_env_float("RISK_DAILY_PROFIT_LOCK_PCT", 0.0) or None)
        if os.environ.get("RISK_DAILY_PROFIT_LOCK_PCT") else None,
        breakeven_trigger_r=_env_float("RISK_BREAKEVEN_TRIGGER_R", 1.0),
        breakeven_buffer_pct=_env_float("RISK_BREAKEVEN_BUFFER_PCT", 0.05),
        partial_close_trigger_r=_env_float("RISK_PARTIAL_CLOSE_TRIGGER_R", 1.0),
        partial_close_fraction=_env_float("RISK_PARTIAL_CLOSE_FRACTION", 0.5),
        trailing_trigger_r=_env_float("RISK_TRAILING_TRIGGER_R", 1.5),
        trailing_distance_r=_env_float("RISK_TRAILING_DISTANCE_R", 0.5),
        use_auto_drawdown_risk=_env_bool("RISK_USE_AUTO_DRAWDOWN", True),
        use_kelly_sizing=_env_bool("RISK_USE_KELLY_SIZING", False),
        use_trailing_stop=_env_bool("RISK_USE_TRAILING_STOP", True),
        max_net_usd_exposure=_env_float("RISK_MAX_NET_USD_EXPOSURE", 2.0),
        kelly_min_trades=_env_int("RISK_KELLY_MIN_TRADES", 10),
        kelly_fraction=_env_float("RISK_KELLY_FRACTION", 0.5),
        min_ai_score=_env_int("RISK_MIN_AI_SCORE", 55),
        min_ml_confidence=_env_float("RISK_MIN_ML_CONFIDENCE", 60.0),
        enable_ml_filter=_env_bool("RISK_ENABLE_ML_FILTER", True),
        max_open_trades=_env_int("RISK_MAX_OPEN_TRADES", 1),
    )

    notifications = NotificationConfig(
        telegram_bot_token=_env_str("TELEGRAM_BOT_TOKEN"),
        telegram_chat_id=_env_str("TELEGRAM_CHAT_ID"),
        discord_webhook_url=_env_str("DISCORD_WEBHOOK_URL"),
        enabled=_env_bool("NOTIFICATIONS_ENABLED", False),
    )

    news = NewsConfig(
        alpha_vantage_api_key=_env_str("ALPHA_VANTAGE_API_KEY"),
        enabled=_env_bool("NEWS_FILTER_ENABLED", False),
    )

    ai = AIConfig(
        provider=_env_str("AI_PROVIDER", "claude"),
        anthropic_api_key=_env_str("ANTHROPIC_API_KEY"),
        google_api_key=_env_str("GOOGLE_API_KEY"),
        openai_api_key=_env_str("OPENAI_API_KEY"),
        model=_env_str("AI_MODEL", "claude-sonnet-5"),
        gemini_model=_env_str("GEMINI_MODEL", "gemini-2.0-flash"),
        gpt_model=_env_str("GPT_MODEL", "gpt-4o-mini"),
        enabled=_env_bool("AI_ASSISTANT_ENABLED", False),
    )

    logging_cfg = LoggingConfig(
        level=_env_str("LOG_LEVEL", "INFO"),
        log_dir=_env_str("LOG_DIR", "logs"),
        log_filename=_env_str("LOG_FILENAME", "trading.log"),
        max_bytes=_env_int("LOG_MAX_BYTES", 5 * 1024 * 1024),
        backup_count=_env_int("LOG_BACKUP_COUNT", 5),
        console=_env_bool("LOG_CONSOLE", True),
    )

    return BotConfig(
        broker=_env_str("BROKER", "paper"),
        mt5=mt5,
        mt5_bridge=mt5_bridge,
        risk=risk,
        news=news,
        ai=ai,
        logging=logging_cfg,
        notifications=notifications,
    )


def load_config(
    overrides: Optional[Dict[str, Any]] = None,
    validate: bool = True,
) -> BotConfig:
    """Build effective BotConfig from environment, then apply overrides.

    Args:
        overrides: Nested dict matching BotConfig shape.
        validate: If True, raises ConfigError on invalid config.

    Returns:
        Populated BotConfig instance.
    """
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    cfg = _config_from_env()

    if overrides:
        merged = _deep_merge(cfg.to_dict(), overrides)
        cfg = BotConfig(
            broker=merged["broker"],
            mt5=MT5Config(**merged["mt5"]),
            mt5_bridge=MT5BridgeConfig(**merged["mt5_bridge"]),
            risk=RiskConfig(**merged["risk"]),
            news=NewsConfig(**merged["news"]),
            ai=AIConfig(**merged["ai"]),
            logging=LoggingConfig(**merged["logging"]),
            notifications=NotificationConfig(**merged["notifications"]),
        )

    if validate:
        cfg.validate()

    return cfg
