"""Data fetching service with caching and fallback.

Provides unified OHLCV data from yfinance (fallback) and MT5 (primary).
"""

from __future__ import annotations

from typing import Optional, Dict, Any
import pandas as pd
import yfinance as yf
from functools import lru_cache

from src.logger import get_logger
from src.exceptions import DataError

log = get_logger(__name__)

# Minutes per candle for each MT5 timeframe string, used for gap detection.
TIMEFRAME_MINUTES: Dict[str, int] = {
    "M1": 1, "M5": 5, "M15": 15, "M30": 30, "H1": 60, "H4": 240, "D1": 1440,
}


def detect_data_gaps(df: pd.DataFrame, timeframe: str, tolerance: float = 3.0) -> Dict[str, Any]:
    """Flag suspicious holes in a historical OHLCV series.

    A "gap" is a jump between consecutive candle timestamps that is more
    than `tolerance` times the expected candle interval. Weekend closures
    (gold/forex markets are closed roughly Friday evening -> Sunday evening)
    are expected and excluded so they aren't reported as data problems.

    Args:
        df: DataFrame with a 'Date' column, already sorted ascending.
        timeframe: MT5 timeframe string (e.g. "H1").
        tolerance: Multiple of the expected interval before a jump counts
            as a gap (default 3x — small jitter shouldn't trigger a warning).

    Returns:
        Dict with gap_count, largest_gap_hours, and a list of (start, end)
        timestamps for each detected gap (capped at 20 for display).
    """
    result: Dict[str, Any] = {"gap_count": 0, "largest_gap_hours": 0.0, "gaps": []}
    if df is None or df.empty or "Date" not in df.columns or len(df) < 2:
        return result

    interval_minutes = TIMEFRAME_MINUTES.get(timeframe, 60)
    expected = pd.Timedelta(minutes=interval_minutes)
    threshold = expected * tolerance

    dates = pd.to_datetime(df["Date"]).reset_index(drop=True)
    deltas = dates.diff().dropna()

    for idx, delta in deltas.items():
        if delta <= threshold:
            continue
        start_ts = dates[idx - 1]
        # Weekend closure: gap starts Fri/Sat and the gap itself is <= ~60h
        # (a normal weekly market close), so don't flag it.
        if start_ts.weekday() >= 4 and delta <= pd.Timedelta(hours=60):
            continue
        result["gap_count"] += 1
        gap_hours = delta.total_seconds() / 3600.0
        result["largest_gap_hours"] = max(result["largest_gap_hours"], gap_hours)
        if len(result["gaps"]) < 20:
            result["gaps"].append((str(start_ts), str(dates[idx])))

    return result


class DataService:
    """Service for fetching and caching market data."""

    def __init__(self, symbol: str = "GC=F") -> None:
        self.symbol = symbol

    @lru_cache(maxsize=8)
    def fetch_yfinance(self, period: str = "5d", interval: str = "15m") -> pd.DataFrame:
        """Fetch data from yfinance with result caching.

        Args:
            period: Data period (e.g., '5d', '1mo', '1y').
            interval: Candle interval (e.g., '15m', '1h', '1d').

        Returns:
            Cleaned DataFrame with standard columns.

        Raises:
            DataError: If fetching or cleaning fails.
        """
        try:
            data = yf.download(self.symbol, period=period, interval=interval, progress=False)
        except Exception as exc:
            raise DataError(f"yfinance fetch failed: {exc}") from exc

        if data is None or data.empty:
            raise DataError("yfinance returned empty data")

        return self._clean_dataframe(data)

    def fetch_from_mt5(self, broker, period: str = "5d", interval: str = "15m") -> pd.DataFrame:
        """Fetch data from MT5 broker if available.

        Args:
            broker: Broker instance with get_ohlcv method.
            period: Ignored for MT5 (uses bars).
            interval: Timeframe string.

        Returns:
            Cleaned DataFrame.
        """
        try:
            df = broker.get_ohlcv(timeframe=interval)
        except Exception as exc:
            raise DataError(f"MT5 data fetch failed: {exc}") from exc

        if df is None or df.empty:
            raise DataError("MT5 returned empty data")

        return self._clean_dataframe(df)

    @staticmethod
    def _clean_dataframe(data: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names and types."""
        df = data.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [' '.join(c).strip() if c[1] not in ('nan', 'NaN') else c[0]
                         for c in df.columns.values]

        if 'Date' not in df.columns and 'Time' not in df.columns:
            df = df.reset_index()

        col_map = {}
        for c in df.columns:
            cs = str(c).lower()
            if 'date' in cs or 'time' in cs:
                col_map[c] = 'Date'
            elif 'open' in cs:
                col_map[c] = 'Open'
            elif 'high' in cs:
                col_map[c] = 'High'
            elif 'low' in cs:
                col_map[c] = 'Low'
            elif 'close' in cs:
                col_map[c] = 'Close'
            elif 'volume' in cs:
                col_map[c] = 'Volume'

        df = df.rename(columns=col_map)
        for col in ('Open', 'High', 'Low', 'Close', 'Volume'):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        return df
