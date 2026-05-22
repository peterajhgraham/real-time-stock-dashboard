"""Spectral analysis of price time-series.

This module is the explicit bridge between the financial signal-processing
work in this repository and frequency-domain analysis of neural / EEG data.
The Welch PSD and STFT spectrogram implemented here are the **same** tools
used to isolate alpha/beta/gamma bands in cortical recordings — see the
honors-thesis ``neural-representation-explorer`` pipeline. Only the labels
on the frequency axis change between domains.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from scipy import signal as sps

from config import PSD_NPERSEG, STFT_NOVERLAP, STFT_NPERSEG, TOP_FREQ_PEAKS


def _detrended(x: np.ndarray) -> np.ndarray:
    """Remove the linear trend before spectral estimation.

    Trend leakage produces spurious low-frequency power; detrending is
    standard practice in both quantitative finance and EEG preprocessing.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 4:
        return x
    return sps.detrend(x, type="linear")


@dataclass
class PSDResult:
    freqs: np.ndarray            # cycles / sample
    power: np.ndarray            # power spectral density
    peak_freqs: np.ndarray       # dominant frequency components
    peak_powers: np.ndarray


def power_spectral_density(
    x: pd.Series,
    fs: float = 1.0,
    nperseg: int = PSD_NPERSEG,
    n_peaks: int = TOP_FREQ_PEAKS,
) -> PSDResult:
    """Welch power-spectral density of the signal.

    Splits the (detrended) signal into overlapping segments, applies a
    Hann window to each, takes the squared magnitude of the FFT, and
    averages across segments. This reduces variance at the cost of
    frequency resolution — the same bias/variance trade-off discussed in
    Bendat & Piersol and applied in EEG band-power estimation.

    Sampling frequency ``fs`` is in cycles-per-sample by default; pass the
    true sampling rate (e.g. 1 / dt seconds) to get physical Hz.
    """
    arr = _detrended(x.to_numpy())
    if arr.size < 8:
        return PSDResult(np.array([]), np.array([]), np.array([]), np.array([]))

    nps = min(nperseg, arr.size)
    freqs, pxx = sps.welch(arr, fs=fs, nperseg=nps, detrend=False, scaling="density")

    # Locate dominant peaks in the spectrum
    peak_idx, _ = sps.find_peaks(pxx)
    if peak_idx.size:
        order = np.argsort(pxx[peak_idx])[::-1][:n_peaks]
        peak_freqs = freqs[peak_idx][order]
        peak_powers = pxx[peak_idx][order]
    else:
        peak_freqs = np.array([])
        peak_powers = np.array([])

    return PSDResult(freqs, pxx, peak_freqs, peak_powers)


@dataclass
class STFTResult:
    times: np.ndarray
    freqs: np.ndarray
    power_db: np.ndarray         # 10*log10 of |STFT|^2


def stft_spectrogram(
    x: pd.Series,
    fs: float = 1.0,
    nperseg: int = STFT_NPERSEG,
    noverlap: int = STFT_NOVERLAP,
) -> STFTResult:
    """Short-time Fourier transform — time-resolved spectral content.

    Slides a windowed FFT across the signal, producing a time x frequency
    energy surface. This is exactly the spectrogram used in EEG analysis
    to track how band power waxes and wanes during a task; in financial
    series it surfaces shifts between calm and volatile regimes that the
    static PSD averages away.
    """
    arr = _detrended(x.to_numpy())
    if arr.size < nperseg:
        nperseg = max(8, arr.size // 2)
        noverlap = nperseg // 2

    freqs, times, zxx = sps.stft(
        arr, fs=fs, nperseg=nperseg, noverlap=noverlap, padded=False, boundary=None
    )
    power = np.abs(zxx) ** 2
    # Avoid log(0); floor to a tiny positive number
    power_db = 10.0 * np.log10(power + 1e-12)
    return STFTResult(times=times, freqs=freqs, power_db=power_db)


def band_power(psd: PSDResult, lo: float, hi: float) -> float:
    """Integrate PSD power between two frequencies.

    Trapezoidal integration of pxx over [lo, hi]. In EEG this returns the
    alpha-band (8-13 Hz) or beta-band (13-30 Hz) power; here it quantifies
    energy in any user-selected frequency window of the price series.
    """
    if psd.freqs.size == 0:
        return float("nan")
    mask = (psd.freqs >= lo) & (psd.freqs <= hi)
    if not np.any(mask):
        return 0.0
    return float(np.trapz(psd.power[mask], psd.freqs[mask]))
