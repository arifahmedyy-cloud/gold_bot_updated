"""Backtesting engine with regime-aware simulation.

Simulates trades over historical data with proper risk management.
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
import pandas as pd
import numpy as np

from src.logger import get_logger
from src.models import BacktestSummary, SignalOutput, TradeSignal
from src.trading.indicators import TechnicalIndicators
from src.trading.regime_detector import RegimeDetector
from src.trading.smc import SMCAnalyzer
from src.trading.decision_engine import DecisionEngine
from src.trading.risk_manager import RiskManager
from src.config import RiskConfig

log = get_logger(__name__)


@dataclass
class SimulatedTrade:
    entry_time: datetime
    exit_time: Optional[datetime] = None
    direction: str = ""
    entry_price: float = 0.0
    exit_price: float = 0.0
    sl: float = 0.0
    tp: float = 0.0
    lot_size: float = 0.0
    profit_loss: float = 0.0
    result: str = ""
    regime: str = ""
    r_multiple: float = 0.0
    duration_hours: float = 0.0
    exit_reason: str = ""


class BacktestEngine:
    """Backtest engine for strategy validation."""

    def __init__(
        self,
        initial_balance: float = 10000.0,
        risk_config: Optional[RiskConfig] = None,
        spread: float = 0.5,
    ) -> None:
        self.initial_balance = initial_balance
        self.risk_config = risk_config or RiskConfig()
        self.spread = spread
        self.equity_curve: List[float] = []
        self.trades: List[SimulatedTrade] = []
        self.regime_detector = RegimeDetector()
        self.smc = SMCAnalyzer()
        self.decision = DecisionEngine(self.risk_config)
        self.risk = RiskManager(self.risk_config)

    def run(
        self,
        df: pd.DataFrame,
        strategy_fn: Optional[Callable[[pd.DataFrame, int], Any]] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> BacktestSummary:
        """Run backtest on historical data.

        Args:
            df: OHLCV DataFrame.
            strategy_fn: Optional custom strategy function. May return either
                a dict (``{"action": ..., "sl": ...}``) or an object with
                those as attributes (e.g. ``strategies.StrategySignal``,
                which is what every built-in strategy in strategies.py
                actually returns).
            progress_callback: Optional callback invoked periodically as
                ``callback(candles_done, candles_total)`` so a caller (e.g.
                a Streamlit progress bar) can reflect run progress.

        Returns:
            BacktestSummary with performance metrics.
        """
        if len(df) < 60:
            raise ValueError(
                f"Not enough historical candles to backtest ({len(df)} given, need at least 60 "
                f"— 50 for indicator warm-up plus room to actually trade). Pick a longer period "
                f"or a lower timeframe."
            )
        # Defensive data hygiene: MT5 history should already arrive sorted
        # and deduplicated, but don't trust that blindly — a single
        # out-of-order or duplicate candle would corrupt SL/TP checks below.
        if "Date" in df.columns:
            df = df.sort_values("Date").drop_duplicates(subset="Date", keep="last").reset_index(drop=True)
        df = TechnicalIndicators.add_all(df)
        balance = self.initial_balance
        peak_balance = balance
        self.equity_curve = [balance]
        self.trades = []
        active_trade: Optional[SimulatedTrade] = None
        total_steps = len(df) - 50

        for i in range(50, len(df)):
            if progress_callback is not None and (i % 25 == 0 or i == len(df) - 1):
                progress_callback(i - 50 + 1, total_steps)
            window = df.iloc[:i+1].copy()
            current = df.iloc[i]
            prev = df.iloc[i-1]

            # Update floating P&L
            if active_trade:
                bid = float(current["Close"])
                ask = bid + self.spread
                if active_trade.direction == "BUY":
                    active_trade.profit_loss = (bid - active_trade.entry_price) * active_trade.lot_size * 100.0
                else:
                    active_trade.profit_loss = (active_trade.entry_price - ask) * active_trade.lot_size * 100.0

                # Check SL/TP
                sl_hit = (
                    (active_trade.direction == "BUY" and bid <= active_trade.sl)
                    or (active_trade.direction == "SELL" and ask >= active_trade.sl)
                )
                tp_hit = (
                    (active_trade.direction == "BUY" and bid >= active_trade.tp)
                    or (active_trade.direction == "SELL" and ask <= active_trade.tp)
                )

                if sl_hit or tp_hit:
                    active_trade.exit_time = pd.Timestamp(current.name) if hasattr(current.name, "year") else datetime.now()
                    active_trade.exit_price = active_trade.sl if sl_hit else active_trade.tp
                    active_trade.result = "LOSS" if sl_hit else "WIN"
                    active_trade.exit_reason = "SL" if sl_hit else "TP"
                    balance += active_trade.profit_loss
                    self.risk.record_trade_result(active_trade.profit_loss)
                    self.trades.append(active_trade)
                    active_trade = None

            # Check daily guard
            guard = self.risk.daily_guard(balance, peak_balance)
            if guard.should_block_new_trades:
                if active_trade and guard.should_close_all:
                    # Close at current price
                    active_trade.exit_time = pd.Timestamp(current.name) if hasattr(current.name, "year") else datetime.now()
                    active_trade.exit_price = float(current["Close"])
                    active_trade.result = "CLOSED_BY_GUARD"
                    active_trade.exit_reason = "Daily guard"
                    balance += active_trade.profit_loss
                    self.risk.record_trade_result(active_trade.profit_loss)
                    self.trades.append(active_trade)
                    active_trade = None
                continue

            # Generate signal
            if strategy_fn:
                sig = strategy_fn(window, i)
                # strategy_fn may return a plain dict OR an object with these
                # as attributes (every built-in strategy in strategies.py
                # returns a StrategySignal dataclass, which has no .get() —
                # calling sig.get(...) on it raised AttributeError and
                # crashed the backtest on the very first candle for every
                # strategy). Handle both shapes.
                def _field(obj: Any, key: str, default: Any) -> Any:
                    if isinstance(obj, dict):
                        return obj.get(key, default)
                    return getattr(obj, key, default)

                raw_action = _field(sig, "action", "NO_TRADE")
                action_value = raw_action if raw_action in ("BUY", "SELL") else "NO_TRADE"
                confidence_value = _field(sig, "confidence", 50)
                signal = SignalOutput(
                    action=action_value,
                    confidence=confidence_value,
                    regime="custom",
                    strategy="Custom",
                    expected_pf=1.5, expected_max_dd=5.0, expected_avg_rr=2.0,
                    consistency_score=confidence_value,
                    sl=_field(sig, "sl", float(current["Close"])),
                    tp=_field(sig, "tp", float(current["Close"])),
                    entry=_field(sig, "entry", float(current["Close"])),
                    lot_size=0.01,
                    explanation="Custom strategy",
                    metrics={},
                )
            else:
                signal = self.regime_detector.generate_signal(window)

            smc_result = self.smc.analyze(window)
            decision = self.decision.decide(signal, smc_result, current_price=float(current["Close"]))

            if decision.action in ("BUY", "SELL") and active_trade is None:
                # Calculate lot size
                risk_pct = self.risk_config.base_risk_pct
                if self.risk_config.use_auto_drawdown_risk:
                    risk_pct = self.risk.compute_drawdown_adjusted_risk(balance, peak_balance, risk_pct)
                if self.risk_config.use_kelly_sizing:
                    kelly_pct, _ = self.risk.kelly_position_size(balance, decision.entry, decision.sl, risk_pct)
                    risk_pct = kelly_pct

                lot_size = self.risk.calculate_lot_size(balance, decision.entry, decision.sl, risk_pct)

                # Validate
                is_valid, reason = self.risk.validate_signal(
                    decision.entry, decision.sl, decision.tp, lot_size,
                    balance, decision.ai_score,
                )
                if not is_valid:
                    continue

                entry_price = decision.entry + self.spread/2 if decision.action == "BUY" else decision.entry - self.spread/2

                active_trade = SimulatedTrade(
                    entry_time=pd.Timestamp(current.name) if hasattr(current.name, "year") else datetime.now(),
                    direction=decision.action,
                    entry_price=entry_price,
                    sl=decision.sl,
                    tp=decision.tp,
                    lot_size=lot_size,
                    regime=signal.regime,
                )

            # Update equity
            floating = active_trade.profit_loss if active_trade else 0.0
            equity = balance + floating
            if equity > peak_balance:
                peak_balance = equity
            self.equity_curve.append(equity)

        # Close any remaining trade
        if active_trade:
            last_close = float(df.iloc[-1]["Close"])
            active_trade.exit_time = datetime.now()
            active_trade.exit_price = last_close
            active_trade.result = "WIN" if active_trade.profit_loss > 0 else "LOSS"
            active_trade.exit_reason = "End of data"
            balance += active_trade.profit_loss
            self.trades.append(active_trade)

        return self._summarize()

    def _summarize(self) -> BacktestSummary:
        wins = [t for t in self.trades if t.result == "WIN"]
        losses = [t for t in self.trades if t.result == "LOSS"]
        total_pl = sum(t.profit_loss for t in self.trades)
        final_balance = self.initial_balance + total_pl
        return_pct = (total_pl / self.initial_balance) * 100 if self.initial_balance > 0 else 0

        win_rate = len(wins) / len(self.trades) * 100 if self.trades else 0
        total_gain = sum(t.profit_loss for t in wins) if wins else 0.01
        total_loss = abs(sum(t.profit_loss for t in losses)) if losses else 0.01
        pf = total_gain / total_loss if total_loss > 0 else float("inf")

        # Max drawdown
        peak = self.initial_balance
        max_dd = 0.0
        for eq in self.equity_curve:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak * 100
            if dd > max_dd:
                max_dd = dd

        # Regime breakdown
        regime_stats: Dict[str, Any] = {}
        for t in self.trades:
            r = t.regime or "unknown"
            if r not in regime_stats:
                regime_stats[r] = {"trades": 0, "wins": 0, "losses": 0, "pnl": 0.0}
            regime_stats[r]["trades"] += 1
            regime_stats[r]["wins"] += 1 if t.result == "WIN" else 0
            regime_stats[r]["losses"] += 1 if t.result == "LOSS" else 0
            regime_stats[r]["pnl"] += t.profit_loss

        return BacktestSummary(
            final_balance=final_balance,
            return_pct=return_pct,
            trades=len(self.trades),
            wins=len(wins),
            win_rate=win_rate,
            profit_factor=pf,
            max_dd=max_dd,
            equity_curve=self.equity_curve,
            regime_breakdown=regime_stats,
        )
