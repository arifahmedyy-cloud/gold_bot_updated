"""Performance reporting and analytics.

Generates comprehensive trade performance statistics.
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import pandas as pd
import numpy as np

from src.logger import get_logger

log = get_logger(__name__)


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics."""
    total_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    avg_drawdown_pct: float = 0.0
    recovery_factor: float = 0.0
    expectancy: float = 0.0
    r_squared: float = 0.0
    monthly_returns: Dict[str, float] = field(default_factory=dict)
    best_month: str = ""
    worst_month: str = ""


class PerformanceReporter:
    """Generate performance reports from trade history."""

    def __init__(self, trades: List[Dict[str, Any]]) -> None:
        self.trades = trades
        self.df = pd.DataFrame(trades) if trades else pd.DataFrame()

    def calculate_metrics(self, starting_balance: float = 10000.0) -> PerformanceMetrics:
        """Calculate all performance metrics."""
        if self.df.empty:
            return PerformanceMetrics()

        wins = self.df[self.df.get("result", "") == "WIN"]
        losses = self.df[self.df.get("result", "") == "LOSS"]

        total = len(self.df)
        win_count = len(wins)
        loss_count = len(losses)
        win_rate = (win_count / total * 100) if total > 0 else 0

        total_gain = wins["profit_loss"].sum() if not wins.empty else 0.01
        total_loss = abs(losses["profit_loss"].sum()) if not losses.empty else 0.01
        pf = total_gain / total_loss if total_loss > 0 else float("inf")

        avg_win = wins["profit_loss"].mean() if not wins.empty else 0.0
        avg_loss = losses["profit_loss"].mean() if not losses.empty else 0.0

        # Consecutive
        results = self.df.get("result", pd.Series()).tolist()
        max_wins = max_losses = current_wins = current_losses = 0
        for r in results:
            if r == "WIN":
                current_wins += 1
                current_losses = 0
                max_wins = max(max_wins, current_wins)
            elif r == "LOSS":
                current_losses += 1
                current_wins = 0
                max_losses = max(max_losses, current_losses)

        # Equity curve
        equity = [starting_balance]
        for pl in self.df.get("profit_loss", []):
            equity.append(equity[-1] + pl)
        equity_arr = np.array(equity)

        # Drawdown
        peak = equity_arr[0]
        max_dd = 0.0
        dd_sum = 0.0
        dd_count = 0
        for eq in equity_arr:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak * 100
            if dd > 0:
                dd_sum += dd
                dd_count += 1
            if dd > max_dd:
                max_dd = dd
        avg_dd = dd_sum / dd_count if dd_count > 0 else 0.0

        # Returns
        returns = np.diff(equity_arr) / equity_arr[:-1]
        if len(returns) > 1 and returns.std() > 0:
            sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
            downside = returns[returns < 0]
            sortino = (returns.mean() / downside.std()) * np.sqrt(252) if len(downside) > 0 else 0.0
        else:
            sharpe = sortino = 0.0

        total_return = (equity_arr[-1] - starting_balance) / starting_balance * 100
        calmar = total_return / max_dd if max_dd > 0 else 0.0
        recovery = total_return / max_dd if max_dd > 0 else 0.0

        # Expectancy
        win_prob = win_count / total if total > 0 else 0
        loss_prob = loss_count / total if total > 0 else 0
        expectancy = (win_prob * avg_win) + (loss_prob * avg_loss) if total > 0 else 0.0

        # Monthly
        if "entry_time" in self.df.columns:
            self.df["month"] = pd.to_datetime(self.df["entry_time"]).dt.to_period("M")
            monthly = self.df.groupby("month")["profit_loss"].sum().to_dict()
            monthly_str = {str(k): float(v) for k, v in monthly.items()}
            best = max(monthly_str, key=monthly_str.get) if monthly_str else ""
            worst = min(monthly_str, key=monthly_str.get) if monthly_str else ""
        else:
            monthly_str = {}
            best = worst = ""

        return PerformanceMetrics(
            total_trades=total,
            win_rate=win_rate,
            profit_factor=pf,
            avg_win=avg_win,
            avg_loss=avg_loss,
            max_consecutive_wins=max_wins,
            max_consecutive_losses=max_losses,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            max_drawdown_pct=max_dd,
            avg_drawdown_pct=avg_dd,
            recovery_factor=recovery,
            expectancy=expectancy,
            r_squared=0.0,
            monthly_returns=monthly_str,
            best_month=best,
            worst_month=worst,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Export metrics as dictionary."""
        metrics = self.calculate_metrics()
        return {
            "total_trades": metrics.total_trades,
            "win_rate": metrics.win_rate,
            "profit_factor": metrics.profit_factor,
            "avg_win": metrics.avg_win,
            "avg_loss": metrics.avg_loss,
            "max_consecutive_wins": metrics.max_consecutive_wins,
            "max_consecutive_losses": metrics.max_consecutive_losses,
            "sharpe_ratio": metrics.sharpe_ratio,
            "sortino_ratio": metrics.sortino_ratio,
            "calmar_ratio": metrics.calmar_ratio,
            "max_drawdown_pct": metrics.max_drawdown_pct,
            "avg_drawdown_pct": metrics.avg_drawdown_pct,
            "recovery_factor": metrics.recovery_factor,
            "expectancy": metrics.expectancy,
            "monthly_returns": metrics.monthly_returns,
            "best_month": metrics.best_month,
            "worst_month": metrics.worst_month,
        }
