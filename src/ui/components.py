"""Reusable Streamlit UI components.

All UI widgets centralized for consistency and maintainability.
"""

from __future__ import annotations

from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
import streamlit as st
import pandas as pd

from src.logger import get_logger
from src.services import credential_store
from src.trading.symbol_manager import SymbolManager

log = get_logger(__name__)


def render_header(title: str = "🥇 XAU/USD AI Trading Bot") -> None:
    """Render page header."""
    st.set_page_config(page_title=title, layout="wide")
    st.title(title)
    st.markdown("---")


def render_sidebar_config() -> Dict[str, Any]:
    """Render configuration sidebar and return settings."""
    with st.sidebar:
        st.header("⚙️ Configuration")

        broker = st.selectbox(
            "Broker", ["paper", "mt5", "mt5_bridge"], index=0,
            help="'mt5' imports the MetaTrader5 package directly — only works if this app "
                 "itself runs on Windows. 'mt5_bridge' talks to a separate Windows-side "
                 "bridge service over HTTP — use this for the Docker/Linux deployment.")

        st.markdown("**Symbols to trade**")
        active_symbols = st.multiselect(
            "Active symbols", options=SymbolManager.all_known_symbols(),
            default=["XAUUSD"],
            help="Gold + Vantage-style forex pairs. Adding more than one enables the "
                 "USD correlation guard to avoid stacking the same USD bet twice.")
        if not active_symbols:
            active_symbols = ["XAUUSD"]

        saved = credential_store.load_credentials(broker) or {}

        with st.expander("MT5 Settings", expanded=(broker == "mt5")):
            mt5_login = st.number_input("Login", value=int(saved.get("login", 0) or 0), step=1, min_value=0)
            mt5_password = st.text_input("Password", type="password", value=saved.get("password", ""))
            mt5_server = st.text_input("Server", value=saved.get("server", ""))
            mt5_leverage = st.number_input("Leverage", value=100, min_value=1, max_value=2000)
            remember_login = st.checkbox(
                "🔒 Remember this login (encrypted, saved on this PC only)",
                value=bool(saved))
            if saved and st.button("Forget saved login"):
                credential_store.clear_credentials(broker)
                st.rerun()

        with st.expander("MT5 Bridge Settings", expanded=(broker == "mt5_bridge")):
            st.caption("Points at the Windows-side bridge service (see mt5_bridge/README.md). "
                       "MT5 login/password/server are configured on the bridge itself, not here.")
            # Reuses credential_store's generic (login, password, server) slots:
            # server holds the bridge URL, password holds the bridge token.
            bridge_saved = credential_store.load_credentials("mt5_bridge") or {}
            bridge_url = st.text_input(
                "Bridge URL", value=bridge_saved.get("server", ""),
                placeholder="http://192.168.1.50:8800")
            bridge_token = st.text_input(
                "Bridge Token", type="password", value=bridge_saved.get("password", ""),
                help="Must match BRIDGE_TOKEN set on the bridge service")
            remember_bridge = st.checkbox(
                "🔒 Remember bridge settings (encrypted, saved on this PC only)",
                value=bool(bridge_saved))
            if bridge_saved and st.button("Forget saved bridge settings"):
                credential_store.clear_credentials("mt5_bridge")
                st.rerun()

        with st.expander("Risk Settings"):
            base_risk = st.slider("Base Risk %", 0.1, 5.0, 1.0, 0.1)
            max_risk = st.slider("Max Risk %", 0.5, 10.0, 2.0, 0.1)
            daily_loss = st.slider("Daily Loss Limit %", 1.0, 20.0, 5.0, 0.5)
            max_trades = st.number_input("Max Open Trades", value=1, min_value=1, max_value=10)
            use_kelly = st.toggle("Use Kelly Sizing", value=False)
            use_drawdown = st.toggle("Auto Drawdown Risk", value=True)

        with st.expander("API Keys (use .env for production)"):
            ai_provider = st.selectbox("AI Reviewer Provider", ["claude", "gemini", "gpt"], index=0)
            anthropic_key = st.text_input("Anthropic API Key", type="password", value="",
                                          help="Leave empty to use .env", disabled=(ai_provider != "claude"))
            google_key = st.text_input("Google (Gemini) API Key", type="password", value="",
                                       help="Leave empty to use .env", disabled=(ai_provider != "gemini"))
            openai_key = st.text_input("OpenAI (GPT) API Key", type="password", value="",
                                       help="Leave empty to use .env", disabled=(ai_provider != "gpt"))
            alpha_vantage_key = st.text_input("Alpha Vantage Key", type="password", value="", help="Leave empty to use .env")
            telegram_token = st.text_input("Telegram Bot Token", type="password", value="")
            telegram_chat = st.text_input("Telegram Chat ID", value="")
            discord_webhook = st.text_input("Discord Webhook URL", type="password", value="")

        st.markdown("---")
        st.caption("v3.0.0 Production | Use .env for secure credential storage")

    return {
        "broker": broker,
        "mt5_login": int(mt5_login),
        "mt5_password": mt5_password,
        "mt5_server": mt5_server,
        "mt5_leverage": float(mt5_leverage),
        "bridge_url": bridge_url,
        "bridge_token": bridge_token,
        "remember_bridge": remember_bridge,
        "active_symbols": active_symbols,
        "remember_login": remember_login,
        "base_risk": float(base_risk),
        "max_risk": float(max_risk),
        "daily_loss": float(daily_loss),
        "max_trades": int(max_trades),
        "use_kelly": use_kelly,
        "use_drawdown": use_drawdown,
        "anthropic_key": anthropic_key,
        "google_key": google_key,
        "openai_key": openai_key,
        "ai_provider": ai_provider,
        "alpha_vantage_key": alpha_vantage_key,
        "telegram_token": telegram_token,
        "telegram_chat": telegram_chat,
        "discord_webhook": discord_webhook,
    }


def render_account_card(account_info: Dict[str, Any]) -> None:
    """Render account information cards."""
    if not account_info:
        st.warning("No account data available")
        return

    cols = st.columns(4)
    metrics = [
        ("💰 Balance", f"${account_info.get('balance', 0):,.2f}"),
        ("📊 Equity", f"${account_info.get('equity', 0):,.2f}"),
        ("📉 Margin", f"${account_info.get('margin', 0):,.2f}"),
        ("📈 Free Margin", f"${account_info.get('free_margin', 0):,.2f}"),
    ]
    for col, (label, value) in zip(cols, metrics):
        col.metric(label, value)


def render_positions_table(positions: List[Dict[str, Any]]) -> None:
    """Render open positions table."""
    if not positions:
        st.info("No open positions")
        return

    df = pd.DataFrame(positions)
    st.dataframe(df, use_container_width=True)


def render_trade_history(trades: List[Dict[str, Any]], limit: int = 50) -> None:
    """Render recent trade history."""
    if not trades:
        st.info("No trade history")
        return

    df = pd.DataFrame(trades[:limit])
    st.dataframe(df, use_container_width=True)


def render_signal_card(signal: Dict[str, Any]) -> None:
    """Render current signal information."""
    if not signal:
        st.info("No active signal")
        return

    action = signal.get("action", "NO_TRADE")
    color = "🟢" if action == "BUY" else "🔴" if action == "SELL" else "⚪"

    st.subheader(f"{color} Signal: {action}")
    cols = st.columns(4)
    cols[0].metric("Entry", f"{signal.get('entry', 0):.2f}")
    cols[1].metric("SL", f"{signal.get('sl', 0):.2f}")
    cols[2].metric("TP", f"{signal.get('tp', 0):.2f}")
    cols[3].metric("Score", f"{signal.get('ai_score', 0)}/100")

    if signal.get("explanation"):
        st.caption(signal["explanation"])


def render_health_status(health: Dict[str, Any]) -> None:
    """Render system health indicators."""
    cols = st.columns(4)
    connected = health.get("connected", False)
    terminal = health.get("terminal_ok", False)
    account = health.get("account_ok", False)

    cols[0].metric("Connection", "✅" if connected else "❌")
    cols[1].metric("Terminal", "✅" if terminal else "❌")
    cols[2].metric("Account", "✅" if account else "❌")
    cols[3].metric("Uptime", f"{health.get('uptime_seconds', 0):.0f}s")


def render_backtest_results(summary: Dict[str, Any]) -> None:
    """Render backtest summary."""
    if not summary:
        return

    st.subheader("📈 Backtest Results")
    cols = st.columns(4)
    cols[0].metric("Final Balance", f"${summary.get('final_balance', 0):,.2f}")
    cols[1].metric("Return", f"{summary.get('return_pct', 0):.2f}%")
    cols[2].metric("Win Rate", f"{summary.get('win_rate', 0):.1f}%")
    cols[3].metric("Max DD", f"{summary.get('max_dd', 0):.2f}%")

    if summary.get("equity_curve"):
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=summary["equity_curve"],
            mode="lines",
            name="Equity",
            line=dict(color="#00ff88"),
        ))
        fig.update_layout(
            title="Equity Curve",
            xaxis_title="Trade",
            yaxis_title="Balance ($)",
            template="plotly_dark",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)


def render_monte_carlo(report: Dict[str, Any]) -> None:
    """Render Monte Carlo simulation results."""
    if not report:
        return

    st.subheader("🎲 Monte Carlo Simulation")
    cols = st.columns(3)
    cols[0].metric("Simulations", report.get("n_simulations", 0))
    cols[1].metric("Risk of Ruin", f"{report.get('risk_of_ruin_pct', 0):.2f}%")
    cols[2].metric("Prob. of Loss", f"{report.get('probability_of_loss_pct', 0):.2f}%")

    stats = report.get("return_stats", {})
    st.write(f"Mean Final Balance: ${stats.get('mean', 0):,.2f}")
    st.write(f"Median: ${stats.get('median', 0):,.2f}")
    st.write(f"95th Percentile: ${stats.get('q95', 0):,.2f}")
    st.write(f"5th Percentile: ${stats.get('q05', 0):,.2f}")


# --- Session windows in UTC. Gold/Forex conventionally split into three
# overlapping sessions; times are approximate market-open hours (not exact
# to the minute, but good enough for a dashboard "which session are we in"
# indicator). Adjust here if your broker's server time differs.
_SESSIONS = [
    ("Sydney/Asian", 22, 7),   # 22:00 UTC -> 07:00 UTC (wraps midnight)
    ("London", 7, 16),         # 07:00 UTC -> 16:00 UTC
    ("New York", 12, 21),      # 12:00 UTC -> 21:00 UTC
]


def render_spread_widget(bid: float, ask: float, warn_threshold: float = 0.5) -> None:
    """Show current bid/ask spread with a warning if it's wide enough to hurt entries."""
    if bid <= 0 or ask <= 0:
        st.caption("Spread: unavailable")
        return
    spread = ask - bid
    wide = spread > warn_threshold
    icon = "🔴" if wide else "🟢"
    st.metric("Spread", f"{spread:.2f}", delta=None)
    if wide:
        st.caption(f"{icon} Spread is wide ({spread:.2f} > {warn_threshold:.2f}) — entries may cost more than expected")
    else:
        st.caption(f"{icon} Spread normal")


def render_session_widget() -> None:
    """Show which trading session(s) are currently active, in UTC."""
    now_utc = datetime.now(timezone.utc)
    hour = now_utc.hour + now_utc.minute / 60.0

    def _in_session(start: int, end: int) -> bool:
        if start < end:
            return start <= hour < end
        return hour >= start or hour < end  # wraps past midnight

    active = [name for name, start, end in _SESSIONS if _in_session(start, end)]
    label = " + ".join(active) if active else "Between sessions (low liquidity)"
    st.metric("Active Session(s)", label)
    st.caption(f"UTC time: {now_utc.strftime('%H:%M')}")
    if not active:
        st.caption("⚪ Low liquidity period — spreads may widen, consider caution")


def render_daily_summary_widget(trades_today: List[Dict[str, Any]]) -> None:
    """Show today's trading summary: P&L, win rate, trade count."""
    if not trades_today:
        st.caption("No trades yet today")
        return
    total_pl = sum(t.get("profit_loss", 0.0) or 0.0 for t in trades_today)
    wins = sum(1 for t in trades_today if (t.get("profit_loss") or 0.0) > 0)
    n = len(trades_today)
    win_rate = (wins / n * 100.0) if n else 0.0

    cols = st.columns(3)
    cols[0].metric("Today's P&L", f"${total_pl:,.2f}", delta=f"{total_pl:+.2f}")
    cols[1].metric("Trades Today", n)
    cols[2].metric("Win Rate Today", f"{win_rate:.0f}%")
