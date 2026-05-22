# Real-Time Time-Series Signal Analysis Platform

A real-time signal visualization and quantitative analysis platform — originally built for financial time-series, the architecture and signal processing primitives apply directly to any streaming time-series data including physiological and neural signals.

The dashboard ingests a live stream of samples, applies a battery of DSP-style filters and statistical estimators, and renders the results — including a Welch power-spectral-density panel and an STFT spectrogram — in a single Streamlit interface. Every indicator is implemented from scratch on top of NumPy / SciPy / Pandas so its filter structure is visible, not hidden inside a black-box library.

*(Originally prototyped as a stock-price toy for my grandmother, who loves investing.)*

![Example dashboard](Example.png)

---

## Signal Processing Framework

Every "technical indicator" in this dashboard is a digital signal processing primitive applied to a one-dimensional time-series. The financial vocabulary is a thin wrapper around standard DSP operators:

| Indicator | DSP interpretation | Formulation |
|---|---|---|
| **SMA** (Simple Moving Average) | Finite impulse response (FIR) box filter — convolution with a uniform 1/N kernel of length N. Linear phase, sinc-shaped magnitude response. | `y[n] = (1/N) · Σ x[n-k]` for `k = 0..N-1` |
| **EMA** (Exponential Moving Average) | First-order infinite impulse response (IIR) filter with an exponential decay kernel. Single real pole; geometric weighting of past samples. | `y[n] = α·x[n] + (1-α)·y[n-1]`, `α = 2/(N+1)` |
| **RSI** (Relative Strength Index) | Bounded momentum oscillator — half-wave rectification of the first difference, followed by Wilder IIR smoothing of the positive and negative branches. | `RS = avg_gain / avg_loss`, `RSI = 100 − 100/(1+RS)` |
| **Bollinger Bands** | Volatility envelope — rolling mean ± k·σ over a sliding window. Statistical tolerance band on the recent local distribution. | `(μ_t − k·σ_t, μ_t + k·σ_t)` |
| **MACD** | Bandpass filter — difference of two EMAs at different time constants isolates a passband between them; the signal line is an EMA of that bandpass output. | `EMA_fast(x) − EMA_slow(x)`, signal = `EMA_sig(MACD)` |
| **ATR** (Average True Range) | Rolling amplitude envelope — Wilder-smoothed true range; tracks instantaneous signal volatility. | `EMA(max(high-low, |high-close₋₁|, |low-close₋₁|))` |
| **OBV** (On-Balance Volume) | Cumulative sign-modulated integrator — discrete analogue of a coulomb counter applied to volume signed by `Δprice`. | `OBV_t = OBV_{t-1} + sign(Δp_t)·v_t` |
| **Rolling z-score** | Local standardisation — mean-centred and scaled by rolling σ so heterogeneous channels can be compared on a common axis. | `z_t = (x_t − μ_t)/σ_t` |
| **Welch PSD** | Power spectral density via Bartlett-windowed segment averaging. The standard estimator for stationary spectral content. | `Ŝ(f) = (1/K) Σ |FFT{x_k · w}|²` |
| **STFT spectrogram** | Short-time Fourier transform — sliding windowed FFT producing a time × frequency energy surface. | `X(t,f) = Σ x[n]·w[n−t]·e^{−j2πfn}` |
| **Autocorrelation** | Sample ACF over lags 1..K; detects serial dependence in returns / increments. | `ρ(k) = Cov(x_t, x_{t-k}) / Var(x_t)` |
| **Rolling Sharpe** | Signal-to-noise ratio of log-returns over a sliding window, annualised. | `√A · μ_W(r) / σ_W(r)` |
| **Drawdown** | Normalised deficit from running maximum — tracks regression below prior peak. | `dd_t = p_t / max_{s≤t} p_s − 1` |
| **HMM regime decoder** | Gaussian hidden Markov model on returns — Viterbi-decoded latent state sequence (bear / sideways / bull). | `p(r_t | s_t) = 𝒩(μ_{s_t}, σ²_{s_t})` |

Treating indicators this way makes their behaviour predictable: SMA introduces a known group delay of `(N−1)/2`, MACD has a passband determined by the two EMA time constants, Bollinger Band breaches are formal outliers under a local Gaussian assumption, etc. It also makes the codebase straightforwardly portable to any other streaming signal.

---

## Extending to Other Domains

Although the dashboard ships with a Yahoo-Finance loader, none of the analysis modules know anything about prices. The exact same pipeline works for:

- **EEG / LFP recordings.** Replace `fetch_stock_data(...)` with an MNE / NWB reader. The Welch PSD becomes a band-power estimator (alpha 8–13 Hz, beta 13–30 Hz, gamma 30+ Hz). The STFT spectrogram is the canonical time-frequency view used in cortical analysis. Bollinger Bands become threshold envelopes for event detection on band-pass-filtered traces.
- **Neural spike rates.** Bin spike times into a rate signal, feed it in as `Close`. SMA produces the trial-averaged firing-rate envelope; EMA gives an online causal estimate. Autocorrelation surfaces refractoriness and rhythmicity. The HMM regime decoder is structurally identical to the behavioural-state decoders used in the honors-thesis [`neural-representation-explorer`](https://github.com/peterajhgraham) — same generative model, different domain.
- **Physiological signals (HRV, EMG, respiration).** RSI's rectify-and-smooth structure is the standard EMG-envelope pipeline. ATR captures rolling amplitude for burst detection. The rolling z-score is the normalisation step before any cross-channel comparison.

**What changes when you swap domains:** the loader, the units on the axes, and the interpretation of dominant frequency peaks.

**What stays the same:** every module under `indicators.py`, `spectral.py`, `stats.py`, and `ui.py`. The DSP is domain-agnostic.

---

## Directory Structure

```
real-time-stock-dashboard/
├── stock_dashboard.py     # Streamlit entry point + data ingestion
├── config.py              # All hyperparameters in one place
├── indicators.py          # SMA, EMA, RSI, Bollinger, MACD, ATR, OBV, z-score
├── spectral.py            # Welch PSD, STFT spectrogram, band-power
├── stats.py               # ACF, returns distribution + KS, Sharpe, drawdown, HMM
├── ui.py                  # Plotly / Streamlit rendering helpers
├── requirements.txt
├── README.md
├── LICENSE
└── Example.png
```

## Features

- **Streaming ingestion** with Yahoo-Finance as the default source; trivially swappable for any other sampled signal.
- **Indicator overlays** on candlestick or line charts: SMA, EMA, Bollinger volatility envelope.
- **Dedicated panels** for MACD (bandpass), ATR, rolling z-score, and OBV.
- **Spectral analysis panel** — Welch PSD with dominant-peak annotation + STFT spectrogram (the explicit bridge to EEG/neural pipelines).
- **Statistical panels** — rolling autocorrelation, returns distribution with normal-fit overlay and KS statistic, rolling annualised Sharpe, drawdown chart.
- **Optional HMM regime decoder** — 3-state Gaussian HMM on log-returns, decoded states drawn as background bands on the main chart.
- **Live sidebar quotes** for a configurable watchlist.

## Installation

```bash
git clone https://github.com/peterajhgraham/real-time-stock-dashboard.git
cd real-time-stock-dashboard
pip3 install -r requirements.txt
python3 -m streamlit run stock_dashboard.py
```

Python 3.9+ is recommended. `hmmlearn` is optional — the HMM regime panel degrades gracefully if it is not installed.

## Usage

Sidebar controls:

- **Ticker** — any symbol supported by Yahoo Finance.
- **Window** — analysis window (`1d` … `max`); the sampling interval is selected automatically.
- **Chart type** — candlestick or line.
- **Overlay indicators** — SMA, EMA, Bollinger Bands superimposed on the price chart.
- **Panels** — toggle MACD, ATR + z-score, OBV, Welch PSD, STFT spectrogram, autocorrelation, returns distribution, rolling Sharpe, drawdown, HMM regimes.

All hyperparameters (window lengths, smoothing constants, FFT segment sizes, HMM seed, colour palette) live in `config.py`.

## Known Issues

- **Data fetching errors:** invalid tickers or rate-limited requests return empty data. The dashboard surfaces a Streamlit error.
- **Short windows:** spectral estimators and the HMM need a reasonable number of samples; they degrade to informational messages rather than crashing.

## Contributing

Contributions welcome — particularly loaders for other signal domains (EEG, NWB, OpenBCI, EDF) so the platform can be exercised on physiological data without modification.

## License

MIT — see the `LICENSE` file.

## Contact

`peter_graham@brown.edu`
