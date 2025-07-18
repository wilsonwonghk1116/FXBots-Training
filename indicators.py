"""
indicators.py
Technical indicators using TA-Lib.
"""
import talib
import numpy as np

def compute_indicators(price_series: np.ndarray) -> dict:
    """Compute common technical indicators using TA-Lib."""
    indicators = {}
    indicators['sma'] = talib.SMA(price_series)
    indicators['ema'] = talib.EMA(price_series)
    indicators['rsi'] = talib.RSI(price_series)
    indicators['macd'], _, _ = talib.MACD(price_series)
    # Add more indicators as needed
    return indicators
# To add new indicators, define a new function or extend compute_indicators. 