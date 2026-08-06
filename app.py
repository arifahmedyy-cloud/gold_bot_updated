"""XAU/USD AI Trading Bot — Production Dashboard."""

from __future__ import annotations

import os, sys, time, threading
from datetime import datetime, timedelta
from typing import Any, List, Dict

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.logger import configure_logging, get_logger
from src.config import load_config, BotConfig
from src.models import TradeSignal
from src.trading.broker_connector import PaperBroker, MT5Broker, MT5BridgeBroker
from src.trading.regime_detector import RegimeDetector
from src.trading.smc import SMCAnalyzer
from src.trading.decision_engine import DecisionEngine
from src.trading.risk_manager import RiskManager
from src.trading.symbol_manager import SymbolManager
from src.trading.correlation_guard import CorrelationGuard, OpenExposure
from src.trading.indicators import TechnicalIndicators
from src.trading.strategies import get_strategy, STRATEGIES, strategy_signal_to_signal_output
from src.trading.trailing_stop_manager import TrailingStopManager
from src.services.data_service import DataService, detect_data_gaps
from src.services.journal_service import JournalService
from src.services.notification_service import NotificationService
from src.services.health_service import HealthService
from src.services.news_service import NewsService
from src.services.ai_service import AIService
from src.services import credential_store
from src.backtesting.backtest_engine import BacktestEngine
from src.backtesting.monte_carlo import MonteCarloSimulator
from src.backtesting.performance_report import PerformanceReporter
from src.ui.components import (
    render_header, render_sidebar_config, render_account_card,
    render_positions_table, render_trade_history, render_signal_card,
    render_health_status, render_backtest_results, render_monte_carlo,
    render_spread_widget, render_session_widget, render_daily_summary_widget,
)

log = get_logger(__name__)


def init_session_state() -> None:
    defaults = {
        "bot_running": False, "last_update": None, "broker": None,
        "data_service": None, "journal": None, "notifier": None,
        "health": None, "regime_detector": None, "smc": None,
        "decision_engine": None, "risk_manager": None,
        "current_signal": None, "backtest_summary": None,
        "monte_carlo_report": None, "peak_balance": 10000.0,
        "error_count": 0, "last_error": None, "broker_signature": None,
        "news_service": None, "ai_service": None,
        "current_news": None, "current_ai": None,
        "symbol_manager": None, "current_signals": {},
        "live_strategy_name": "Regime (default)", "symbol_signal_cache": {},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def create_broker(config: BotConfig) -> Any:
    if config.broker == "paper":
        return PaperBroker(initial_balance=10000.0, leverage=config.mt5.leverage, symbol="XAUUSD")
    elif config.broker == "mt5":
        return MT5Broker(
            login=config.mt5.login, password=config.mt5.password,
            server=config.mt5.server, leverage=config.mt5.leverage,
            reconnect_attempts=config.mt5.reconnect_attempts,
            reconnect_backoff_seconds=config.mt5.reconnect_backoff_seconds,
            reconnect_backoff_multiplier=config.mt5.reconnect_backoff_multiplier,
        )
    elif config.broker == "mt5_bridge":
        # App runs on Linux/Docker; MT5 itself is reached over HTTP via a
        # small Windows-side bridge service (see mt5_bridge/). This is the
        # broker to use in the Docker deployment described in the README.
        return MT5BridgeBroker(
            base_url=config.mt5_bridge.base_url, token=config.mt5_bridge.token,
            symbol_candidates=config.mt5_bridge.symbol_candidates,
            reconnect_attempts=config.mt5_bridge.reconnect_attempts,
            reconnect_backoff_seconds=config.mt5_bridge.reconnect_backoff_seconds,
            reconnect_backoff_multiplier=config.mt5_bridge.reconnect_backoff_multiplier,
            request_timeout_seconds=config.mt5_bridge.request_timeout_seconds,
            leverage=config.mt5_bridge.leverage,
        )
    raise ValueError(f"Unknown broker: {config.broker}")


YFINANCE_TICKER_MAP = {
    "XAUUSD": "GC=F", "EURUSD": "EURUSD=X", "GBPUSD": "GBPUSD=X",
    "USDJPY": "JPY=X", "AUDUSD": "AUDUSD=X",
}


def _yf_ticker(symbol: str) -> str:
    return YFINANCE_TICKER_MAP.get(symbol, f"{symbol}=X")


def _run_symbol_cycle(
    symbol: str, config: BotConfig, broker: Any, journal: Any, notifier: Any, health: Any,
    regime: Any, smc: Any, decision: Any, risk: Any, news_service: Any, ai_service: Any,
    symbol_manager: Any, correlation_guard: Any, account: Any, live_strategy_name: str,
    signal_cache: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Run one full analysis+trade cycle for a single symbol. Returns the
    signal dict for display. Never raises — caller's try/except wraps this,
    but errors here are also caught locally so one bad symbol doesn't stop
    the others in the same cycle.

    Indicators/regime-or-strategy/SMC are only recomputed when a new candle
    is detected (candle-based caching) — `signal_cache` persists this across
    calls (owned by trading_loop, one entry per symbol) so repeated 3-second
    auto-refresh cycles reuse the same analysis until the market actually
    produces a new bar.
    """
    try:
        if config.broker == "mt5" and broker.is_connected():
            df = broker.get_ohlcv(symbol=symbol, timeframe="H1", bars=200)
        else:
            df = DataService(_yf_ticker(symbol)).fetch_yfinance(period="5d", interval="15m")
        if df is None or df.empty:
            log.warning("No data received for %s", symbol)
            return {}

        health.record_market_data_update()

        candle_time = df.index[-1]
        cache_key = (candle_time, live_strategy_name)
        cached = signal_cache.get(symbol)

        if cached is not None and cached.get("cache_key") == cache_key:
            df_ind = cached["df_ind"]
            regime_signal = cached["regime_signal"]
            smc_result = cached["smc_result"]
            current_price = cached["current_price"]
        else:
            df_ind = TechnicalIndicators.add_all(df)
            current_price = float(df_ind.iloc[-1]["Close"])
            smc_result = smc.analyze(df_ind)

            if live_strategy_name and live_strategy_name != "Regime (default)":
                strategy_signal = get_strategy(live_strategy_name).generate(df_ind)
                regime_signal = strategy_signal_to_signal_output(strategy_signal, live_strategy_name)
            else:
                regime_signal = regime.generate_signal(df_ind)

            signal_cache[symbol] = {
                "cache_key": cache_key, "df_ind": df_ind, "regime_signal": regime_signal,
                "smc_result": smc_result, "current_price": current_price,
            }

        strategy_label = live_strategy_name if live_strategy_name and live_strategy_name != "Regime (default)" else "Regime+SMC"

        profile = symbol_manager.profile(symbol)
        try:
            price_info = broker.get_price(symbol)
            spread = (price_info.get("ask", 0) - price_info.get("bid", 0)) / profile.pip_size
            if spread > profile.max_spread_pips:
                log.info("%s spread too wide (%.1f pips), skipping this cycle", symbol, spread)
                return {"action": "NO_TRADE", "entry": current_price, "sl": 0, "tp": 0,
                        "ai_score": 0, "regime": regime_signal.regime, "smc_bias": smc_result.bias,
                        "explanation": f"Spread too wide ({spread:.1f} pips)"}
        except Exception:
            pass  # spread check is best-effort; don't block trading on it

        news_sentiment = None
        if news_service is not None:
            news_sentiment = news_service.fetch_news_sentiment(symbol)

        ml_confidence = None
        ai_result = None
        if ai_service is not None and ai_service.is_available:
            ai_result = ai_service.analyze(regime_signal, smc_result, news_sentiment, current_price)
            if ai_result.available:
                ml_confidence = ai_result.confidence

        final_decision = decision.decide(
            regime_signal, smc_result, ml_confidence=ml_confidence, current_price=current_price)

        signal_dict = {
            "symbol": symbol, "action": final_decision.action, "entry": final_decision.entry,
            "sl": final_decision.sl, "tp": final_decision.tp,
            "ai_score": final_decision.ai_score, "regime": regime_signal.regime,
            "smc_bias": smc_result.bias, "explanation": final_decision.explanation,
        }

        if final_decision.action in ("BUY", "SELL"):
            positions = broker.get_positions()
            if len(positions) >= config.risk.max_open_trades:
                log.info("Max open trades reached (%d)", len(positions))
                return signal_dict

            open_exposure = [OpenExposure(symbol=p.symbol, direction=p.position_type) for p in positions]
            corr_result = correlation_guard.check(open_exposure, symbol, final_decision.action)
            if not corr_result.allowed:
                log.info("Correlation guard blocked %s %s: %s", final_decision.action, symbol, corr_result.reason)
                signal_dict["explanation"] = corr_result.reason
                signal_dict["action"] = "NO_TRADE"
                return signal_dict

            risk_pct = config.risk.base_risk_pct
            if config.risk.use_auto_drawdown_risk:
                risk_pct = risk.compute_drawdown_adjusted_risk(
                    account.balance, st.session_state.peak_balance, risk_pct)
            if config.risk.use_kelly_sizing:
                kelly_pct, _ = risk.kelly_position_size(
                    account.balance, final_decision.entry, final_decision.sl, risk_pct)
                risk_pct = kelly_pct

            lot_size = risk.calculate_lot_size(
                account.balance, final_decision.entry, final_decision.sl,
                risk_pct, leverage=account.leverage)

            is_valid, reason = risk.validate_signal(
                final_decision.entry, final_decision.sl, final_decision.tp,
                lot_size, account.balance, final_decision.ai_score)
            if not is_valid:
                log.warning("Signal rejected for %s: %s", symbol, reason)
                return signal_dict

            signal = TradeSignal(
                symbol=symbol, direction=final_decision.action,
                entry_price=final_decision.entry, sl=final_decision.sl,
                tp=final_decision.tp, lot_size=lot_size,
                strategy=strategy_label, ai_score=final_decision.ai_score,
                regime=regime_signal.regime)
            result = broker.send_order(signal)
            if result["success"]:
                journal.record_order_open(
                    ticket=result["ticket"], entry_time=datetime.now(),
                    symbol=symbol, direction=final_decision.action,
                    entry_price=final_decision.entry, stop_loss=final_decision.sl,
                    take_profit=final_decision.tp, lot_size=lot_size,
                    risk_pct=risk_pct, ai_score=final_decision.ai_score,
                    regime=regime_signal.regime, smc_bias=smc_result.bias,
                    confluence_notes=" | ".join(final_decision.confluence_notes),
                    strategy=strategy_label)
                journal.record_position_risk(
                    ticket=result["ticket"], symbol=symbol, direction=final_decision.action,
                    entry_price=final_decision.entry, initial_sl=final_decision.sl)
                health.record_order(action=final_decision.action, symbol=symbol,
                                    success=True, ticket=result["ticket"])
                if notifier:
                    notifier.notify_trade(final_decision.action, final_decision.entry,
                                          final_decision.sl, final_decision.tp, lot_size)
            else:
                health.record_order(action=final_decision.action, symbol=symbol,
                                    success=False, reason=result.get("error"))

        return signal_dict
    except Exception as exc:
        log.error("Cycle error for %s: %s", symbol, exc, exc_info=True)
        return {}


def _apply_trailing_stops(config: BotConfig, broker: Any, journal: Any, trailing_mgr: Any) -> None:
    """Check every open position for a trailing-stop/break-even update.

    Runs once per trading-loop iteration (not per symbol) since it needs to
    see all open positions together. Never raises — a failure here should
    not stop the trading loop.
    """
    if not config.risk.use_trailing_stop:
        return
    try:
        positions = broker.get_positions()
        open_tickets = {p.ticket for p in positions}

        # Garbage-collect risk-reference rows for positions that are no
        # longer open (closed by SL/TP or manually) so the table doesn't
        # grow unbounded.
        for ticket in journal.all_position_risk_tickets():
            if ticket not in open_tickets:
                journal.clear_position_risk(ticket)

        for pos in positions:
            risk_ref = journal.get_position_risk(pos.ticket)
            if not risk_ref:
                continue  # opened before this feature existed, or by another tool — skip safely
            try:
                price_info = broker.get_price(pos.symbol)
                current_price = price_info["bid"] if pos.position_type == "SELL" else price_info["ask"]
            except Exception:
                continue

            result = trailing_mgr.compute_new_sl(
                direction=pos.position_type, entry_price=risk_ref["entry_price"],
                initial_sl=risk_ref["initial_sl"], current_price=current_price, current_sl=pos.sl,
            )
            if result.new_sl is not None:
                ok = broker.modify_position_sl_tp(pos.ticket, sl=result.new_sl)
                if ok:
                    log.info("Trailing stop updated #%d %s: SL -> %.5f (%s)",
                             pos.ticket, pos.symbol, result.new_sl, result.reason)
    except Exception as exc:
        log.error("Trailing stop check failed: %s", exc, exc_info=True)


def trading_loop(config: BotConfig) -> None:
    broker = st.session_state.broker
    if broker is None:
        log.error("Broker not initialized")
        return
    journal = st.session_state.journal
    notifier = st.session_state.notifier
    health = st.session_state.health
    regime = st.session_state.regime_detector
    smc = st.session_state.smc
    decision = st.session_state.decision_engine
    risk = st.session_state.risk_manager
    news_service = st.session_state.news_service
    ai_service = st.session_state.ai_service
    symbol_manager = st.session_state.symbol_manager
    correlation_guard = CorrelationGuard(symbol_manager, max_net_usd_exposure=config.risk.max_net_usd_exposure)
    trailing_mgr = TrailingStopManager(config.risk)

    while st.session_state.bot_running:
        try:
            account = broker.get_account_info()
            if account.balance > st.session_state.peak_balance:
                st.session_state.peak_balance = account.balance

            guard = risk.daily_guard(account.balance, st.session_state.peak_balance)
            if guard.should_block_new_trades:
                if guard.should_close_all:
                    broker.close_all_positions()
                    health.record_error(guard.reason)
                time.sleep(10)
                continue

            _apply_trailing_stops(config, broker, journal, trailing_mgr)

            signals_this_cycle = {}
            for symbol in symbol_manager.active_symbols:
                signal_dict = _run_symbol_cycle(
                    symbol, config, broker, journal, notifier, health, regime, smc,
                    decision, risk, news_service, ai_service, symbol_manager,
                    correlation_guard, account, st.session_state.live_strategy_name,
                    st.session_state.symbol_signal_cache,
                )
                if signal_dict:
                    signals_this_cycle[symbol] = signal_dict

            if signals_this_cycle:
                st.session_state.current_signals = signals_this_cycle
                primary = symbol_manager.active_symbols[0]
                if primary in signals_this_cycle:
                    st.session_state.current_signal = signals_this_cycle[primary]

            st.session_state.last_update = datetime.now()
            st.session_state.error_count = 0
            time.sleep(3)
        except Exception as exc:
            log.error("Trading loop error: %s", exc, exc_info=True)
            st.session_state.error_count += 1
            st.session_state.last_error = str(exc)
            health.record_error(str(exc))
            if st.session_state.error_count > 5:
                log.critical("Too many errors, stopping bot")
                st.session_state.bot_running = False
            time.sleep(5)




def preflight_check(config: BotConfig, broker: Any) -> List[Dict[str, Any]]:
    """Run once before allowing 'Start Bot'. Returns a list of check results.

    Each item: {"label": str, "ok": bool, "critical": bool, "detail": str}
    Critical checks must all pass before the bot is allowed to start.
    Non-critical checks (AI/news) only show a warning — the bot can still
    run without them, just with reduced analysis.
    """
    checks = []

    connected = False
    try:
        connected = broker.is_connected()
    except Exception:
        pass
    checks.append({"label": "Broker connection", "ok": connected, "critical": True,
                    "detail": f"{config.broker.upper()} " + ("connected" if connected else "not connected")})

    valid_risk = 0 < config.risk.base_risk_pct <= config.risk.max_risk_pct
    checks.append({"label": "Risk settings", "ok": valid_risk, "critical": True,
                    "detail": "base risk % must be > 0 and ≤ max risk %"})

    data_ok = False
    try:
        df = DataService("GC=F").fetch_yfinance(period="1d", interval="5m")
        data_ok = df is not None and not df.empty
    except Exception:
        data_ok = False
    checks.append({"label": "Market data", "ok": data_ok, "critical": True,
                    "detail": "price data reachable" if data_ok else "could not fetch price data"})

    ai_ok = st.session_state.ai_service is not None and st.session_state.ai_service.is_available
    checks.append({"label": "AI reviewer", "ok": ai_ok, "critical": False,
                    "detail": "active" if ai_ok else "not configured — bot will trade on rules only"})

    news_ok = config.news.enabled and bool(config.news.alpha_vantage_api_key)
    checks.append({"label": "News sentiment", "ok": news_ok, "critical": False,
                    "detail": "active" if news_ok else "not configured"})

    return checks


def main() -> None:
    render_header()
    init_session_state()
    ui_config = render_sidebar_config()

    try:
        config = load_config(overrides={
            "broker": ui_config["broker"],
            "mt5": {"login": ui_config["mt5_login"], "password": ui_config["mt5_password"],
                    "server": ui_config["mt5_server"], "leverage": ui_config["mt5_leverage"]},
            "mt5_bridge": {"base_url": ui_config["bridge_url"], "token": ui_config["bridge_token"]},
            "risk": {"base_risk_pct": ui_config["base_risk"], "max_risk_pct": ui_config["max_risk"],
                     "daily_loss_limit_pct": ui_config["daily_loss"], "max_open_trades": ui_config["max_trades"],
                     "use_kelly_sizing": ui_config["use_kelly"], "use_auto_drawdown_risk": ui_config["use_drawdown"]},
            "notifications": {"telegram_bot_token": ui_config["telegram_token"],
                              "telegram_chat_id": ui_config["telegram_chat"],
                              "discord_webhook_url": ui_config["discord_webhook"],
                              "enabled": bool(ui_config["telegram_token"] or ui_config["discord_webhook"])},
            **({"ai": {"provider": ui_config["ai_provider"],
                       "anthropic_api_key": ui_config["anthropic_key"],
                       "google_api_key": ui_config["google_key"],
                       "openai_api_key": ui_config["openai_key"],
                       "enabled": True}}
               if (ui_config["anthropic_key"] or ui_config["google_key"] or ui_config["openai_key"])
               else {}),
            **({"news": {"alpha_vantage_api_key": ui_config["alpha_vantage_key"], "enabled": True}}
               if ui_config["alpha_vantage_key"] else {}),
        }, validate=False)
    except Exception as exc:
        st.error(f"Config error: {exc}")
        return

    broker_signature = (config.broker, config.mt5.login, config.mt5.password,
                        config.mt5.server, config.mt5.leverage,
                        config.mt5_bridge.base_url, config.mt5_bridge.token,
                        config.ai.provider, config.ai.anthropic_api_key,
                        config.ai.google_api_key, config.ai.openai_api_key, config.ai.enabled,
                        config.news.alpha_vantage_api_key, config.news.enabled,
                        tuple(ui_config.get("active_symbols") or ["XAUUSD"]))

    if st.session_state.broker is None or st.session_state.broker_signature != broker_signature:
        if st.session_state.broker is not None:
            try:
                st.session_state.broker.disconnect()
            except Exception:
                pass
            if st.session_state.health is not None:
                st.session_state.health.stop()

        try:
            broker = create_broker(config)
            connected = broker.connect()
            if not connected:
                st.error(f"❌ {config.broker.upper()} connection FAILED — check login/password/server and try again.")
                st.session_state.broker = None
                st.session_state.broker_signature = None
                return
            st.session_state.broker = broker
            st.session_state.broker_signature = broker_signature
            st.session_state.journal = JournalService()
            st.session_state.notifier = NotificationService(config.notifications)
            st.session_state.health = HealthService(broker, interval_seconds=30, notifier=st.session_state.notifier)
            st.session_state.regime_detector = RegimeDetector()
            st.session_state.smc = SMCAnalyzer()
            st.session_state.decision_engine = DecisionEngine(config.risk)
            st.session_state.risk_manager = RiskManager(config.risk)
            st.session_state.news_service = NewsService(config.news)
            st.session_state.ai_service = AIService(config.ai)
            st.session_state.symbol_manager = SymbolManager(ui_config.get("active_symbols") or ["XAUUSD"])
            st.success(f"✅ {config.broker.upper()} connected successfully")
            if config.broker == "mt5":
                if ui_config.get("remember_login"):
                    credential_store.save_credentials(
                        "mt5", str(config.mt5.login), config.mt5.password, config.mt5.server)
                else:
                    credential_store.clear_credentials("mt5")
            if config.broker == "mt5_bridge":
                if ui_config.get("remember_bridge"):
                    credential_store.save_credentials(
                        "mt5_bridge", "", config.mt5_bridge.token, config.mt5_bridge.base_url)
                else:
                    credential_store.clear_credentials("mt5_bridge")
        except Exception as exc:
            st.error(f"Initialization failed: {exc}")
            log.error("Init error: %s", exc, exc_info=True)
            st.session_state.broker = None
            st.session_state.broker_signature = None
            return

    broker = st.session_state.broker
    journal = st.session_state.journal
    health = st.session_state.health

    st.caption(f"🔌 Broker: **{config.broker.upper()}** | Connection status: "
               f"{'🟢 Connected' if broker.is_connected() else '🔴 Disconnected'}")

    ai_ready = st.session_state.ai_service is not None and st.session_state.ai_service.is_available
    news_ready = config.news.enabled and bool(config.news.alpha_vantage_api_key)
    risk_ready = 0 < config.risk.base_risk_pct <= config.risk.max_risk_pct
    st.caption(
        f"{'🟢' if broker.is_connected() else '🔴'} Broker &nbsp; "
        f"{'🟢' if risk_ready else '🔴'} Risk config &nbsp; "
        f"{'🟢' if ai_ready else '⚪'} AI reviewer &nbsp; "
        f"{'🟢' if news_ready else '⚪'} News sentiment &nbsp; "
        f"{'🟢' if not st.session_state.bot_running or st.session_state.error_count == 0 else '🟡'} No recent errors",
        unsafe_allow_html=True,
    )

    tabs = st.tabs(["📊 Dashboard", "📈 Backtest", "🎲 Monte Carlo", "📋 Journal", "⚙️ Settings"])

    with tabs[0]:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader("Account Overview")
            try:
                account = broker.get_account_info()
                render_account_card({"balance": account.balance, "equity": account.equity,
                                     "margin": account.margin, "free_margin": account.free_margin})
            except Exception as exc:
                st.error(f"Account info error: {exc}")
            st.subheader("Open Positions")
            try:
                positions = broker.get_positions()
                render_positions_table([
                    {"ticket": p.ticket, "symbol": p.symbol, "type": p.position_type,
                     "volume": p.volume, "open_price": p.open_price, "sl": p.sl, "tp": p.tp, "profit": p.profit}
                    for p in positions])
            except Exception as exc:
                st.error(f"Positions error: {exc}")
            st.subheader("Current Signal")
            render_signal_card(st.session_state.current_signal)

            st.subheader("📰 News Sentiment & 🤖 AI Reviewer")
            ai_col1, ai_col2 = st.columns(2)
            with ai_col1:
                news = st.session_state.current_news
                if news:
                    label_emoji = {"bullish": "🟢", "bearish": "🔴", "neutral": "⚪"}.get(news["label"], "⚪")
                    st.markdown(f"**News:** {label_emoji} {news['label'].title()} "
                                f"(score: {news['score']:.2f}, {news['article_count']} articles)")
                    if news["source"] == "none":
                        st.caption("Alpha Vantage key not set — news sentiment disabled")
                    for h in news["headlines"][:3]:
                        st.caption(f"• {h}")
                else:
                    st.caption("No news data yet — starts once the bot runs a cycle")
            with ai_col2:
                ai = st.session_state.current_ai
                if ai:
                    if ai["available"]:
                        st.markdown(f"**AI Reviewer confidence** ({st.session_state.ai_service.active_provider_name if st.session_state.ai_service else 'n/a'}): {ai['confidence']:.0f}/100")
                        st.progress(min(max(ai["confidence"] / 100, 0.0), 1.0))
                        st.caption(ai["reasoning"])
                    else:
                        st.caption(f"AI reviewer unavailable: {ai.get('error', 'not configured')}")
                else:
                    st.caption("Add an Anthropic API key in the sidebar to enable the AI reviewer")

            if st.button("🔍 Run Analysis Now", help="Fetch data and run one news + AI review cycle without starting the bot"):
                with st.spinner("Analyzing..."):
                    try:
                        df_now = DataService("GC=F").fetch_yfinance(period="5d", interval="15m")
                        if df_now is None or df_now.empty:
                            st.warning("No market data available")
                        else:
                            df_ind_now = TechnicalIndicators.add_all(df_now)
                            regime_now = st.session_state.regime_detector.generate_signal(df_ind_now)
                            smc_now = st.session_state.smc.analyze(df_ind_now)
                            price_now = float(df_ind_now.iloc[-1]["Close"])

                            news_now = None
                            if st.session_state.news_service is not None:
                                news_now = st.session_state.news_service.fetch_news_sentiment("XAUUSD")
                                st.session_state.current_news = {
                                    "score": news_now.score, "label": news_now.label,
                                    "article_count": news_now.article_count,
                                    "headlines": news_now.headlines, "source": news_now.source,
                                }
                            if st.session_state.ai_service is not None and st.session_state.ai_service.is_available:
                                ai_now = st.session_state.ai_service.analyze(regime_now, smc_now, news_now, price_now)
                                st.session_state.current_ai = {
                                    "confidence": ai_now.confidence, "reasoning": ai_now.reasoning,
                                    "available": ai_now.available, "error": ai_now.error,
                                }
                            else:
                                st.session_state.current_ai = {
                                    "confidence": None, "reasoning": "", "available": False,
                                    "error": "AI reviewer not configured",
                                }
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Analysis failed: {exc}")

            st.subheader("Price Chart")

            @st.fragment(run_every="5s")
            def _live_chart():
                try:
                    df = DataService("GC=F").fetch_yfinance(period="1d", interval="5m")
                    if df.empty:
                        st.warning("No chart data available")
                        return
                    fig = go.Figure(data=[go.Candlestick(
                        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
                        name="XAU/USD")])

                    try:
                        chart_start = df.index.min()
                        closed_trades = [t for t in journal.trades
                                          if t.get("entry_time") and pd.Timestamp(t["entry_time"]) >= chart_start]
                        open_trades = [t for t in journal.open_orders
                                       if t.get("entry_time") and pd.Timestamp(t["entry_time"]) >= chart_start]
                        for label, trades, symbol_shape in (
                            ("Closed entry", closed_trades, None), ("Open entry", open_trades, None)
                        ):
                            buys = [t for t in trades if t.get("direction") == "BUY"]
                            sells = [t for t in trades if t.get("direction") == "SELL"]
                            if buys:
                                fig.add_trace(go.Scatter(
                                    x=[pd.Timestamp(t["entry_time"]) for t in buys],
                                    y=[t["entry_price"] for t in buys],
                                    mode="markers", name=f"{label} BUY",
                                    marker=dict(symbol="triangle-up", size=12, color="lime")))
                            if sells:
                                fig.add_trace(go.Scatter(
                                    x=[pd.Timestamp(t["entry_time"]) for t in sells],
                                    y=[t["entry_price"] for t in sells],
                                    mode="markers", name=f"{label} SELL",
                                    marker=dict(symbol="triangle-down", size=12, color="red")))
                    except Exception as marker_exc:
                        log.debug("Trade marker overlay skipped: %s", marker_exc)

                    fig.update_layout(template="plotly_dark", height=500,
                                       margin=dict(l=10, r=10, t=30, b=10))
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption(f"🔄 Auto-refreshes every 5s | Last update: {datetime.now().strftime('%H:%M:%S')}")
                except Exception as exc:
                    st.warning(f"Chart unavailable: {exc}")

            _live_chart()
        with col2:
            st.subheader("Bot Control")
            live_strategy_options = ["Regime (default)"] + STRATEGIES
            selected_strategy = st.selectbox(
                "Live Strategy", live_strategy_options,
                index=live_strategy_options.index(st.session_state.live_strategy_name)
                if st.session_state.live_strategy_name in live_strategy_options else 0,
                disabled=st.session_state.bot_running,
                help="'Regime (default)' uses the built-in RegimeDetector (recommended). "
                     "Choosing a specific strategy replaces it for live trading — SMC, risk, "
                     "AI review, and correlation checks still run the same either way.")
            if selected_strategy != st.session_state.live_strategy_name:
                st.session_state.live_strategy_name = selected_strategy
                st.session_state.symbol_signal_cache = {}  # invalidate cache on strategy switch
            if st.button("▶️ Start Bot", disabled=st.session_state.bot_running):
                with st.spinner("Running pre-flight checks..."):
                    checks = preflight_check(config, broker)
                critical_failed = [c for c in checks if c["critical"] and not c["ok"]]
                for c in checks:
                    icon = "🟢" if c["ok"] else ("🔴" if c["critical"] else "🟡")
                    st.caption(f"{icon} {c['label']}: {c['detail']}")
                if critical_failed:
                    st.error("❌ Cannot start — fix the issue(s) marked 🔴 above first.")
                else:
                    st.session_state.bot_running = True
                    health.start()
                    threading.Thread(target=trading_loop, args=(config,), daemon=True).start()
                    st.success("Bot started")
                    st.rerun()
            if st.button("⏹️ Stop Bot", disabled=not st.session_state.bot_running):
                st.session_state.bot_running = False
                health.stop()
                st.warning("Bot stopped")
                st.rerun()

            st.markdown("---")
            st.subheader("Market Context")
            try:
                price = broker.get_price("XAUUSD")
                render_spread_widget(price.get("bid", 0.0), price.get("ask", 0.0))
            except Exception as exc:
                st.caption(f"Spread unavailable: {exc}")
            render_session_widget()

            st.markdown("---")
            st.subheader("Today's Summary")
            try:
                today_str = datetime.now().strftime("%Y-%m-%d")
                trades_today = [t for t in journal.trades
                                 if t.get("entry_time") and str(t["entry_time"]).startswith(today_str)]
                render_daily_summary_widget(trades_today)
            except Exception as exc:
                st.caption(f"Summary unavailable: {exc}")

            st.markdown("---")
            st.subheader("Health Status")
            try:
                status = health.latest_status()
                render_health_status({"connected": status.connected, "terminal_ok": status.terminal_ok,
                                      "account_ok": status.account_ok, "uptime_seconds": status.uptime_seconds})
            except Exception as exc:
                st.error(f"Health error: {exc}")
            st.markdown("---")
            st.subheader("Risk Status")
            risk = st.session_state.risk_manager
            if risk:
                summary = risk.get_risk_summary()
                st.write(f"Daily P&L: ${summary['daily_pnl']:.2f}")
                st.write(f"Consecutive Losses: {summary['consecutive_losses']}")
                st.write(f"Total Trades: {summary['total_trades']}")
                st.write(f"Win Rate: {summary['win_rate']*100:.1f}%")
            if st.session_state.last_error:
                st.error(f"Last Error: {st.session_state.last_error[:100]}")

    with tabs[1]:
        st.subheader("Strategy Backtest")
        strategy_name = st.selectbox("Strategy", STRATEGIES)
        bt_timeframe = st.selectbox("Timeframe", ["M1", "M5", "M15", "H1", "H4", "D1"], index=3)
        period = st.selectbox("Period", ["1mo", "3mo", "6mo", "1y"], index=1)
        initial_balance = st.number_input("Initial Balance", value=10000.0, step=1000.0)
        bt_symbol = getattr(broker, "resolved_symbol", None) or "XAUUSD"
        st.caption(f"Data source: MT5 ({config.broker.upper()}) | Symbol: {bt_symbol}")

        if st.button("Run Backtest"):
            if config.broker not in ("mt5", "mt5_bridge") or not broker.is_connected():
                st.error(
                    "❌ Backtest needs a connected MT5 broker. Select 'mt5' (or 'mt5_bridge') "
                    "in the sidebar and connect first — 'paper' mode has no historical data source."
                )
            else:
                period_days = {"1mo": 30, "3mo": 90, "6mo": 180, "1y": 365}[period]
                end = datetime.now()
                start = end - timedelta(days=period_days)
                progress = st.progress(0, text=f"Fetching {bt_symbol} {bt_timeframe} history from MT5...")
                df = None
                try:
                    df = broker.load_historical_data(start=start, end=end, symbol=bt_symbol, timeframe=bt_timeframe)
                except Exception as exc:
                    progress.empty()
                    st.error(f"❌ MT5 historical data fetch failed for {bt_symbol} {bt_timeframe}: {exc}")
                    log.error("Backtest data fetch failed: %s", exc, exc_info=True)

                if df is not None and df.empty:
                    progress.empty()
                    st.error(
                        f"❌ MT5 returned no candles for {bt_symbol} {bt_timeframe} over the last {period}. "
                        f"Common causes: this symbol/timeframe's history isn't downloaded in your MT5 "
                        f"terminal yet (open its chart in MT5 once to force a download), or the symbol name "
                        f"doesn't match your broker (try XAUUSDm / XAUUSD.a / GOLD)."
                    )
                elif df is not None:
                    gaps = detect_data_gaps(df, bt_timeframe)
                    if gaps["gap_count"] > 0:
                        st.warning(
                            f"⚠️ {gaps['gap_count']} data gap(s) found in the fetched history "
                            f"(largest ≈ {gaps['largest_gap_hours']:.1f}h). Backtest will still run, "
                            f"but results spanning those gaps may be less reliable."
                        )
                    progress.progress(20, text=f"Loaded {len(df)} candles. Running backtest...")
                    try:
                        strategy = get_strategy(strategy_name)
                        engine = BacktestEngine(initial_balance=initial_balance, risk_config=config.risk)

                        def _bt_progress(done: int, total: int) -> None:
                            pct = 20 + int(done / max(total, 1) * 80)
                            progress.progress(min(pct, 100), text=f"Simulating candle {done}/{total}...")

                        summary = engine.run(df, strategy_fn=strategy.generate, progress_callback=_bt_progress)
                        progress.empty()
                        st.session_state.backtest_summary = {
                            "final_balance": summary.final_balance, "return_pct": summary.return_pct,
                            "trades": summary.trades, "wins": summary.wins, "win_rate": summary.win_rate,
                            "profit_factor": summary.profit_factor, "max_dd": summary.max_dd,
                            "equity_curve": summary.equity_curve, "regime_breakdown": summary.regime_breakdown}
                        render_backtest_results(st.session_state.backtest_summary)
                    except Exception as exc:
                        progress.empty()
                        st.error(f"❌ Backtest simulation failed: {exc}")
                        log.error("Backtest run failed for %s %s: %s", bt_symbol, bt_timeframe, exc, exc_info=True)
        elif st.session_state.backtest_summary:
            render_backtest_results(st.session_state.backtest_summary)

    with tabs[2]:
        st.subheader("Monte Carlo Simulation")
        n_sims = st.slider("Simulations", 100, 5000, 1000, 100)
        ruin_threshold = st.slider("Ruin Threshold %", 10, 90, 50, 5)
        if st.button("Run Monte Carlo"):
            with st.spinner("Running simulation..."):
                try:
                    trades = journal.trades
                    returns = [t.get("profit_loss", 0) for t in trades if t.get("profit_loss") is not None]
                    if len(returns) < 10:
                        st.warning("Need at least 10 trades for meaningful simulation")
                    else:
                        report = MonteCarloSimulator(n_simulations=n_sims).run(returns, ruin_threshold_pct=ruin_threshold)
                        st.session_state.monte_carlo_report = {
                            "n_simulations": report.n_simulations, "risk_of_ruin_pct": report.risk_of_ruin_pct,
                            "probability_of_loss_pct": report.probability_of_loss_pct,
                            "return_stats": report.return_stats, "worst_drawdown_pct": report.worst_drawdown_pct,
                            "average_drawdown_pct": report.average_drawdown_pct,
                            "confidence_intervals": report.confidence_intervals}
                        render_monte_carlo(st.session_state.monte_carlo_report)
                except Exception as exc:
                    st.error(f"Monte Carlo failed: {exc}")
        elif st.session_state.monte_carlo_report:
            render_monte_carlo(st.session_state.monte_carlo_report)

    with tabs[3]:
        st.subheader("Trade Journal")
        trades = journal.trades
        render_trade_history(trades, limit=100)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Export CSV"):
                st.download_button("Download CSV", journal.to_csv(trades), "trades.csv", "text/csv")
        with col2:
            if st.button("Export JSON"):
                st.download_button("Download JSON", journal.to_json(trades), "trades.json", "application/json")
        st.subheader("Performance Analytics")
        if trades:
            st.json(PerformanceReporter(trades).to_dict())
        else:
            st.info("No trades recorded yet")

    with tabs[4]:
        st.subheader("Configuration")
        st.json(config.to_safe_dict())
        st.subheader("Logs")
        log_dir = config.logging.log_dir
        if os.path.exists(log_dir):
            for f in sorted(os.listdir(log_dir))[-5:]:
                st.text(f)


if __name__ == "__main__":
    configure_logging(level="INFO", console=True)
    main()
