"""Central configuration for the time-series signal analysis platform.

All tunable parameters (window sizes, smoothing constants, plotting colours)
live here so individual modules stay declarative. Treat this file as the
single source of truth for analysis hyperparameters.
"""

from dataclasses import dataclass, field
from typing import Dict, List


# ---------------------------------------------------------------------------
# Indicator windows (samples)
# ---------------------------------------------------------------------------
SMA_WINDOW: int = 20            # FIR box-filter length
EMA_WINDOW: int = 20            # IIR exponential-decay time constant
RSI_WINDOW: int = 14            # Wilder smoothing window for RSI
BOLLINGER_WINDOW: int = 20      # Volatility-envelope rolling window
BOLLINGER_K: float = 2.0        # Width of the envelope in standard deviations
MACD_FAST: int = 12             # Fast EMA in MACD bandpass
MACD_SLOW: int = 26             # Slow EMA in MACD bandpass
MACD_SIGNAL: int = 9            # EMA of the MACD line itself
ATR_WINDOW: int = 14            # True-range smoothing window
ZSCORE_WINDOW: int = 60         # Rolling normalisation window


# ---------------------------------------------------------------------------
# Spectral analysis
# ---------------------------------------------------------------------------
PSD_NPERSEG: int = 128          # Welch segment length for power-spectral density
STFT_NPERSEG: int = 64          # STFT window length for the spectrogram
STFT_NOVERLAP: int = 48         # STFT overlap (samples)
TOP_FREQ_PEAKS: int = 5         # Number of dominant peaks to annotate


# ---------------------------------------------------------------------------
# Statistical panels
# ---------------------------------------------------------------------------
AUTOCORR_MAX_LAG: int = 20
SHARPE_WINDOW: int = 30
ANNUALISATION_FACTOR: float = 252.0   # trading days per year
RETURNS_HIST_BINS: int = 50


# ---------------------------------------------------------------------------
# Regime detection (Hidden Markov Model on returns)
# ---------------------------------------------------------------------------
HMM_N_STATES: int = 3           # bull / sideways / bear
HMM_N_ITER: int = 200
HMM_RANDOM_SEED: int = 42


# ---------------------------------------------------------------------------
# UI / plotting palette
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Palette:
    price: str = "#1f77b4"
    sma: str = "#ff7f0e"
    ema: str = "#2ca02c"
    upper_band: str = "#9467bd"
    lower_band: str = "#9467bd"
    macd: str = "#17becf"
    signal: str = "#e377c2"
    histogram: str = "#7f7f7f"
    regime_colors: List[str] = field(default_factory=lambda: ["#d62728", "#7f7f7f", "#2ca02c"])


PALETTE = Palette()


# ---------------------------------------------------------------------------
# Default tickers shown in the sidebar live-quote section
# ---------------------------------------------------------------------------
DEFAULT_SYMBOLS: List[str] = ["AAPL", "GOOGL", "AMZN", "MSFT"]


# Yahoo-Finance period -> recommended sampling interval
INTERVAL_MAPPING: Dict[str, str] = {
    "1d": "1m",
    "5d": "5m",
    "1mo": "1h",
    "3mo": "1d",
    "6mo": "1d",
    "1y": "1wk",
    "5y": "1mo",
    "max": "1mo",
}
