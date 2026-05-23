"""Technical indicators implemented as digital-signal-processing primitives.

Each function is written from scratch with NumPy/Pandas so the underlying
filter structure is explicit. The docstrings describe (a) the mathematical
formulation, and (b) the analogous role the same operator plays in
physiological / neural signal processing - the math is identical, only the
domain interpretation changes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import (
    ATR_WINDOW,
    BOLLINGER_K,
    BOLLINGER_WINDOW,
    EMA_WINDOW,
    MACD_FAST,
    MACD_SIGNAL,
    MACD_SLOW,
    RSI_WINDOW,
    SMA_WINDOW,
    ZSCORE_WINDOW,
)


def sma(x: pd.Series, window: int = SMA_WINDOW) -> pd.Series:
    """Simple Moving Average - FIR box filter, uniform 1/N kernel.

    y[n] = (1/N) * sum_{k=0}^{N-1} x[n-k]

    Equivalent to convolution with a rectangular window of length N. Acts as
    a low-pass filter with linear phase but a sinc-shaped magnitude response.
    In neural data this is the same operator used to compute trial-averaged
    firing-rate envelopes from binned spike counts.
    """
    return x.rolling(window=window, min_periods=window).mean()


def ema(x: pd.Series, window: int = EMA_WINDOW) -> pd.Series:
    """Exponential Moving Average - first-order IIR low-pass filter.

    y[n] = alpha * x[n] + (1 - alpha) * y[n-1],  alpha = 2 / (N + 1)

    Geometric (exponentially-decaying) kernel; emphasises recent samples and
    has a single real pole. In EEG pipelines the same recursive smoother is
    used for online estimation of band-power envelopes.
    """
    return x.ewm(span=window, adjust=False).mean()


def rsi(x: pd.Series, window: int = RSI_WINDOW) -> pd.Series:
    """Relative Strength Index - bounded momentum oscillator on first differences.

    Computes the ratio of Wilder-smoothed positive vs. negative first
    differences of the signal, mapped to [0, 100]:

        RS = avg_gain / avg_loss
        RSI = 100 - 100 / (1 + RS)

    Operationally this is a non-linear rectification of dx/dt followed by an
    IIR smoother - directly analogous to the rectify-and-smooth EMG envelope
    pipeline used to estimate muscle activation.
    """
    diff = x.diff()
    gain = diff.clip(lower=0.0)
    loss = -diff.clip(upper=0.0)
    # Wilder smoothing == EMA with alpha = 1/N
    avg_gain = gain.ewm(alpha=1.0 / window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


def bollinger_bands(
    x: pd.Series,
    window: int = BOLLINGER_WINDOW,
    k: float = BOLLINGER_K,
) -> pd.DataFrame:
    """Bollinger Bands - rolling mean +/- k * sigma volatility envelope.

    Returns columns ``mid``, ``upper``, ``lower``. Equivalent to a moving
    estimate of the signal's local mean and standard deviation; samples
    outside the envelope are statistical outliers relative to the recent
    distribution. In neural recordings the same construction yields
    threshold envelopes for spike detection on band-pass-filtered traces.
    """
    mid = x.rolling(window=window, min_periods=window).mean()
    sd = x.rolling(window=window, min_periods=window).std(ddof=0)
    return pd.DataFrame({"mid": mid, "upper": mid + k * sd, "lower": mid - k * sd})


def macd(
    x: pd.Series,
    fast: int = MACD_FAST,
    slow: int = MACD_SLOW,
    signal: int = MACD_SIGNAL,
) -> pd.DataFrame:
    """MACD - difference of two EMAs, a bandpass filter on the signal.

    macd_line   = EMA_fast(x) - EMA_slow(x)
    signal_line = EMA_signal(macd_line)
    histogram   = macd_line - signal_line

    Subtracting two low-pass IIRs of different time constants produces a
    bandpass response whose passband sits between the two cutoff
    frequencies. This is structurally identical to the Difference-of-
    Gaussians and Difference-of-Exponentials filters used to isolate
    physiological rhythms (e.g. theta vs. delta bands in LFP).
    """
    fast_ema = ema(x, fast)
    slow_ema = ema(x, slow)
    line = fast_ema - slow_ema
    sig = ema(line, signal)
    return pd.DataFrame({"macd": line, "signal": sig, "histogram": line - sig})


def average_true_range(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = ATR_WINDOW,
) -> pd.Series:
    """Average True Range - Wilder-smoothed rolling volatility measure.

    TR_t = max(high - low,
               |high - close_{t-1}|,
               |low  - close_{t-1}|)
    ATR  = EMA_Wilder(TR, window)

    Captures local amplitude of the signal envelope, robust to gaps.
    Analogous to the rolling RMS amplitude used to gate physiological
    bursts (e.g. ripple detection in hippocampal LFP).
    """
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / window, adjust=False).mean()


def on_balance_volume(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume - cumulative volume signed by price first-difference.

    OBV_t = OBV_{t-1} + sign(close_t - close_{t-1}) * volume_t

    A running integral of a sign-modulated input - the discrete analogue of
    a coulomb-counting integrator. In neural analysis the same construction
    builds cumulative spike-count drift signals from sign-coded events.
    """
    direction = np.sign(close.diff().fillna(0.0))
    return (direction * volume).cumsum()


def rolling_zscore(x: pd.Series, window: int = ZSCORE_WINDOW) -> pd.Series:
    """Rolling z-score - local standardisation of the signal.

    z_t = (x_t - mu_t) / sigma_t

    where mu_t and sigma_t are computed over the trailing ``window`` samples.
    Removes slow drift and unit-dependent scaling so heterogeneous channels
    can be compared on a common axis. This is the same normalisation step
    applied to multi-electrode neural data before population-level analysis.
    """
    mu = x.rolling(window=window, min_periods=window).mean()
    sd = x.rolling(window=window, min_periods=window).std(ddof=0)
    return (x - mu) / sd


def add_all_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """Attach every indicator to a price DataFrame in one pass."""
    out = data.copy()
    close = out["Close"]

    out["SMA"] = sma(close)
    out["EMA"] = ema(close)
    out["RSI"] = rsi(close)

    bb = bollinger_bands(close)
    out["BB_MID"] = bb["mid"]
    out["BB_UPPER"] = bb["upper"]
    out["BB_LOWER"] = bb["lower"]

    m = macd(close)
    out["MACD"] = m["macd"]
    out["MACD_SIGNAL"] = m["signal"]
    out["MACD_HIST"] = m["histogram"]

    out["ATR"] = average_true_range(out["High"], out["Low"], close)
    if "Volume" in out.columns:
        out["OBV"] = on_balance_volume(close, out["Volume"])
    out["ZSCORE"] = rolling_zscore(close)
    return out
