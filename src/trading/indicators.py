"""Technical Indicators Module.

All indicators return Series that can be added to a DataFrame.
Optimized to avoid unnecessary copies.
"""

from __future__ import annotations

import pandas as pd
import numpy as np


class TechnicalIndicators:
    """Calculate technical indicators for regime detection and signals."""

    @staticmethod
    def add_all(data: pd.DataFrame) -> pd.DataFrame:
        """Add all indicators to dataframe in one call.

        Args:
            data: Raw OHLCV DataFrame.

        Returns:
            DataFrame with all indicator columns added.
        """
        df = data.copy()

        # Moving Averages
        for period in (10, 20, 50, 200):
            df[f'SMA_{period}'] = df['Close'].rolling(period).mean()
            df[f'EMA_{period}'] = df['Close'].ewm(span=period, adjust=False).mean()

        # RSI
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # MACD
        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema12 - ema26
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

        # Bollinger Bands
        df['BB_Mid'] = df['Close'].rolling(20).mean()
        bb_std = df['Close'].rolling(20).std()
        df['BB_Upper'] = df['BB_Mid'] + 2 * bb_std
        df['BB_Lower'] = df['BB_Mid'] - 2 * bb_std
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Mid']
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])

        # ATR
        tr1 = df['High'] - df['Low']
        tr2 = (df['High'] - df['Close'].shift(1)).abs()
        tr3 = (df['Low'] - df['Close'].shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['ATR_14'] = tr.rolling(14).mean()
        df['ATR_Pct'] = (df['ATR_14'] / df['Close']) * 100

        # ADX
        plus_dm = df['High'].diff().clip(lower=0)
        minus_dm = (-df['Low'].diff()).clip(lower=0)
        plus_dm = plus_dm.where(plus_dm > minus_dm, 0)
        minus_dm = minus_dm.where(minus_dm > plus_dm, 0)
        atr = tr.rolling(14).mean()
        plus_di = 100 * plus_dm.rolling(14).mean() / atr
        minus_di = 100 * minus_dm.rolling(14).mean() / atr
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
        df['ADX'] = dx.rolling(14).mean()
        df['DI_Plus'] = plus_di
        df['DI_Minus'] = minus_di

        # Volatility
        df['Volatility'] = df['Close'].pct_change().rolling(20).std() * np.sqrt(252)

        # Trend Strength Score (0-100)
        df['Trend_Score'] = (
            (df['EMA_20'] > df['EMA_50']).astype(int) * 25 +
            (df['Close'] > df['EMA_20']).astype(int) * 25 +
            (df['ADX'] > 25).astype(int) * 25 +
            (df['MACD_Hist'] > 0).astype(int) * 25
        )

        return df
