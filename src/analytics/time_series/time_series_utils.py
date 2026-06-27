"""
Time Series utilities.
Ported from weg-statistics-python/app/time_series/
"""
import math
import numpy as np
import pandas as pd


def decompose_series(series: pd.Series, period: int, model: str = 'additive') -> dict:
    """
    Decomposes a time series into trend, seasonal, residual, observed.
    model: 'additive' or 'multiplicative'
    Returns dict with arrays: trend, seasonal, residual, observed (same length as series).
    NaN values are preserved (edges of trend).
    """
    from statsmodels.tsa.seasonal import seasonal_decompose
    result = seasonal_decompose(series, model=model, period=period, extrapolate_trend='freq')
    return {
        'observed': series.to_numpy(dtype=float),
        'trend': result.trend.to_numpy(dtype=float),
        'seasonal': result.seasonal.to_numpy(dtype=float),
        'residual': result.resid.to_numpy(dtype=float),
    }


def adf_test(series: pd.Series, max_lag: int = None) -> dict:
    """
    Augmented Dickey-Fuller test for stationarity.
    Returns dict with: statistic, p_value, n_lags, critical_values, is_stationary
    """
    from statsmodels.tsa.stattools import adfuller
    s = series.dropna()
    kw = {'regression': 'c'}
    if max_lag is not None:
        kw['maxlag'] = max_lag
    result = adfuller(s, **kw)
    return {
        'statistic': float(result[0]),
        'p_value': float(result[1]),
        'n_lags': int(result[2]),
        'n_obs': int(result[3]),
        'critical_values': {k: float(v) for k, v in result[4].items()},
        'is_stationary': bool(result[1] < 0.05),
    }


def infer_period(series: pd.Series, index=None) -> int:
    """Heuristic: if index is datetime with monthly freq return 12, quarterly 4, else 12."""
    if index is not None and hasattr(index, 'freq') and index.freq is not None:
        freq_name = str(index.freq)
        if 'M' in freq_name or 'BM' in freq_name:
            return 12
        if 'Q' in freq_name:
            return 4
        if 'W' in freq_name:
            return 52
        if 'D' in freq_name:
            return 7
    return 12
