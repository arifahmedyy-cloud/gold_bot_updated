"""Monte Carlo simulation for risk assessment.

Simulates thousands of equity curves from trade distribution.
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

from src.logger import get_logger
from src.models import MonteCarloReport

log = get_logger(__name__)


class MonteCarloSimulator:
    """Run Monte Carlo simulations on trade history."""

    def __init__(self, n_simulations: int = 1000, seed: Optional[int] = None) -> None:
        self.n_simulations = n_simulations
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)

    def run(
        self,
        returns: List[float],
        starting_balance: float = 10000.0,
        ruin_threshold_pct: float = 50.0,
    ) -> MonteCarloReport:
        """Run Monte Carlo simulation.

        Args:
            returns: List of trade returns (percentage or absolute).
            starting_balance: Initial account balance.
            ruin_threshold_pct: Drawdown threshold for ruin calculation.

        Returns:
            MonteCarloReport with statistical analysis.
        """
        if not returns:
            log.warning("No returns provided for Monte Carlo")
            return MonteCarloReport(n_simulations=0, method="none", starting_balance=starting_balance)

        returns_arr = np.array(returns)
        n_trades = len(returns_arr)
        sim_size = n_trades

        # Run simulations
        final_balances = []
        max_drawdowns = []
        sample_curves: Dict[str, List[float]] = {}

        for sim in range(self.n_simulations):
            shuffled = np.random.choice(returns_arr, size=sim_size, replace=True)
            equity = [starting_balance]
            peak = starting_balance
            max_dd = 0.0
            for ret in shuffled:
                new_eq = equity[-1] + ret
                equity.append(new_eq)
                if new_eq > peak:
                    peak = new_eq
                dd = (peak - new_eq) / peak * 100
                if dd > max_dd:
                    max_dd = dd

            final_balances.append(equity[-1])
            max_drawdowns.append(max_dd)

            if sim < 5:
                sample_curves[f"sim_{sim+1}"] = equity

        final_balances_arr = np.array(final_balances)
        max_dd_arr = np.array(max_drawdowns)

        # Statistics
        return_stats = {
            "mean": float(np.mean(final_balances_arr)),
            "median": float(np.median(final_balances_arr)),
            "std": float(np.std(final_balances_arr)),
            "min": float(np.min(final_balances_arr)),
            "max": float(np.max(final_balances_arr)),
            "q05": float(np.percentile(final_balances_arr, 5)),
            "q25": float(np.percentile(final_balances_arr, 25)),
            "q75": float(np.percentile(final_balances_arr, 75)),
            "q95": float(np.percentile(final_balances_arr, 95)),
        }

        # Confidence intervals
        confidence_intervals = {
            "80%": {
                "lower": float(np.percentile(final_balances_arr, 10)),
                "upper": float(np.percentile(final_balances_arr, 90)),
            },
            "90%": {
                "lower": float(np.percentile(final_balances_arr, 5)),
                "upper": float(np.percentile(final_balances_arr, 95)),
            },
            "95%": {
                "lower": float(np.percentile(final_balances_arr, 2.5)),
                "upper": float(np.percentile(final_balances_arr, 97.5)),
            },
        }

        # Risk metrics
        probability_of_loss = np.mean(final_balances_arr < starting_balance) * 100
        ruin_threshold = starting_balance * (1 - ruin_threshold_pct / 100)
        risk_of_ruin = np.mean(final_balances_arr < ruin_threshold) * 100

        return MonteCarloReport(
            n_simulations=self.n_simulations,
            method="bootstrap",
            starting_balance=starting_balance,
            seed=self.seed,
            n_trades=n_trades,
            sim_size=sim_size,
            returns_pct=[float(r) for r in returns_arr],
            final_balances=[float(b) for b in final_balances],
            return_stats=return_stats,
            max_dd_pct_per_sim=[float(d) for d in max_dd_arr],
            worst_drawdown_pct=float(np.max(max_dd_arr)),
            average_drawdown_pct=float(np.mean(max_dd_arr)),
            confidence_intervals=confidence_intervals,
            probability_of_loss_pct=float(probability_of_loss),
            risk_of_ruin_pct=float(risk_of_ruin),
            ruin_threshold_pct=ruin_threshold_pct,
            sample_equity_curves=sample_curves,
        )
