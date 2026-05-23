"""Real-Time Time-Series Signal Analysis Platform.

A streaming visualisation and quantitative-analysis dashboard. The data
source happens to be Yahoo Finance equity prices, but every analysis
primitive (FIR/IIR filters, PSD, STFT, autocorrelation, HMM regimes) is
domain-agnostic - swap the loader and the same pipeline runs on EEG,
spike-rate envelopes, or any other streaming time-series.
"""

from __future__ import annotations

import streamlit as st
import yfinance as yf

from config import DEFAULT_SYMBOLS, INTERVAL_MAPPING
from indicators import add_all_indicators
from spectral import power_spectral_density, stft_spectrogram
from stats import (
    autocorrelation,
    drawdown,
    fit_regime_hmm,
    log_returns,
    returns_distribution,
    rolling_sharpe,
)
from ui import (
    build_price_figure,
    render_autocorrelation_panel,
    render_drawdown_panel,
    render_macd_panel,
    render_obv_panel,
    render_psd_panel,
    render_regime_summary,
    render_returns_distribution_panel,
    render_sharpe_panel,
    render_spectrogram_panel,
    render_volatility_panel,
)


# ---------------------------------------------------------------------------
# Data ingestion
# ---------------------------------------------------------------------------
def fetch_stock_data(ticker: str, period: str, interval: str):
    try:
        data = yf.download(ticker, period=period, interval=interval, progress=False)
        if data is None or data.empty:
            st.error(f"No data found for {ticker}. Check the ticker and try again.")
            return None
        if hasattr(data.columns, "nlevels") and data.columns.nlevels > 1:
            data.columns = data.columns.get_level_values(0)
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None


@st.cache_data(ttl=60)
def fetch_realtime_data(symbol: str):
    return fetch_stock_data(symbol, "1d", "1m")


def process_data(data):
    """Localise the timestamp index and flatten it into a column."""
    if data.index.tzinfo is None:
        data.index = data.index.tz_localize("UTC")
    data.index = data.index.tz_convert("US/Eastern")
    data = data.reset_index()
    if "Date" in data.columns:
        data = data.rename(columns={"Date": "Datetime"})
    return data


def calculate_metrics(data):
    last_close = float(data["Close"].iloc[-1])
    prev_close = float(data["Close"].iloc[0])
    change = last_close - prev_close
    pct_change = (change / prev_close) * 100.0
    high = float(data["High"].max())
    low = float(data["Low"].min())
    volume = int(data["Volume"].sum()) if "Volume" in data.columns else 0
    return last_close, change, pct_change, high, low, volume


# ---------------------------------------------------------------------------
# App layout
# ---------------------------------------------------------------------------
st.set_page_config(layout="wide", page_title="Time-Series Signal Analysis Platform")
st.title("Real-Time Time-Series Signal Analysis Platform")
st.caption(
    "Streaming financial data shown here, but the indicators, spectral tools "
    "and statistical panels apply directly to any time-series signal - EEG, "
    "spike-rate envelopes, physiological recordings."
)

# Sidebar -- analysis parameters
st.sidebar.header("Signal Parameters")
ticker = st.sidebar.text_input("Ticker", "AAPL")
time_period = st.sidebar.selectbox(
    "Window", ["1d", "5d", "1mo", "3mo", "6mo", "1y", "5y", "max"], index=3
)
chart_type = st.sidebar.selectbox("Chart Type", ["Candlestick", "Line"])
overlays = st.sidebar.multiselect(
    "Overlay indicators",
    ["SMA", "EMA", "Bollinger Bands"],
    default=["SMA", "Bollinger Bands"],
)

st.sidebar.subheader("Panels")
show_macd = st.sidebar.checkbox("MACD (bandpass)", value=True)
show_vol = st.sidebar.checkbox("ATR + rolling z-score", value=True)
show_obv = st.sidebar.checkbox("On-Balance Volume", value=True)
show_psd = st.sidebar.checkbox("Welch PSD", value=True)
show_stft = st.sidebar.checkbox("STFT spectrogram", value=True)
show_acf = st.sidebar.checkbox("Autocorrelation", value=True)
show_dist = st.sidebar.checkbox("Returns distribution + KS", value=True)
show_sharpe = st.sidebar.checkbox("Rolling Sharpe", value=True)
show_dd = st.sidebar.checkbox("Drawdown", value=True)
show_regime = st.sidebar.checkbox("HMM regimes (stretch)", value=False)

run = st.sidebar.button("Update")


if run:
    raw = fetch_stock_data(ticker, time_period, INTERVAL_MAPPING[time_period])
    if raw is not None:
        data = process_data(raw)
        data = add_all_indicators(data)

        last_close, change, pct_change, high, low, volume = calculate_metrics(data)
        st.metric(
            label=f"{ticker} last price",
            value=f"{last_close:.2f} USD",
            delta=f"{change:.2f} ({pct_change:.2f}%)",
        )
        c1, c2, c3 = st.columns(3)
        c1.metric("High", f"{high:.2f} USD")
        c2.metric("Low", f"{low:.2f} USD")
        c3.metric("Volume", f"{volume:,}")

        # ------------------------------------------------------------------
        # Optional regime decoding -- needed before the main chart for shading
        # ------------------------------------------------------------------
        regime = fit_regime_hmm(data["Close"]) if show_regime else None

        st.subheader("Price + indicators")
        st.plotly_chart(
            build_price_figure(data, chart_type, overlays, regime=regime),
            use_container_width=True,
        )

        if show_regime:
            if regime is None:
                st.info(
                    "Regime decoding skipped: install `hmmlearn` or supply a "
                    "longer window."
                )
            else:
                st.markdown(
                    "Background bands show HMM-decoded regimes "
                    "(bear / sideways / bull). Same generative model is used "
                    "for behavioural-state decoding in neural data."
                )
                render_regime_summary(regime)

        # ------------------------------------------------------------------
        # Indicator panels
        # ------------------------------------------------------------------
        if show_macd:
            render_macd_panel(data)
        if show_vol:
            render_volatility_panel(data)
        if show_obv:
            render_obv_panel(data)

        # ------------------------------------------------------------------
        # Spectral analysis -- the bridge to EEG / neural pipelines
        # ------------------------------------------------------------------
        if show_psd or show_stft:
            st.subheader("Spectral analysis")
            st.caption(
                "Welch PSD and STFT are the same tools used to detect EEG "
                "frequency bands (alpha, beta, gamma). Only the meaning of "
                "the frequency axis changes between domains."
            )
        if show_psd:
            psd = power_spectral_density(data["Close"])
            render_psd_panel(psd)
        if show_stft:
            stft = stft_spectrogram(data["Close"])
            render_spectrogram_panel(stft)

        # ------------------------------------------------------------------
        # Statistical panels
        # ------------------------------------------------------------------
        if show_acf:
            ac = autocorrelation(log_returns(data["Close"]).dropna())
            render_autocorrelation_panel(ac)
        if show_dist:
            rd = returns_distribution(data["Close"])
            render_returns_distribution_panel(rd)
        if show_sharpe:
            render_sharpe_panel(rolling_sharpe(data["Close"]), data)
        if show_dd:
            render_drawdown_panel(drawdown(data["Close"]), data)

        # ------------------------------------------------------------------
        # Raw tables
        # ------------------------------------------------------------------
        with st.expander("Historical samples"):
            cols = [c for c in ["Datetime", "Open", "High", "Low", "Close", "Volume"]
                    if c in data.columns]
            st.dataframe(data[cols])
        with st.expander("Indicator table"):
            cols = [c for c in ["Datetime", "SMA", "EMA", "RSI",
                                "BB_UPPER", "BB_LOWER",
                                "MACD", "MACD_SIGNAL", "MACD_HIST",
                                "ATR", "OBV", "ZSCORE"] if c in data.columns]
            st.dataframe(data[cols])


# ---------------------------------------------------------------------------
# Sidebar live quotes
# ---------------------------------------------------------------------------
st.sidebar.header("Live quotes")
for symbol in DEFAULT_SYMBOLS:
    rt = fetch_realtime_data(symbol)
    if rt is not None:
        rt = process_data(rt)
        last_price = float(rt["Close"].iloc[-1])
        open_price = float(rt["Open"].iloc[0])
        delta = last_price - open_price
        pct = (delta / open_price) * 100.0 if open_price else 0.0
        st.sidebar.metric(symbol, f"{last_price:.2f} USD", f"{delta:.2f} ({pct:.2f}%)")

st.sidebar.subheader("About")
st.sidebar.info(
    "Real-time time-series signal analysis platform. Indicators implemented "
    "as DSP primitives; spectral and statistical panels are domain-agnostic "
    "and extend directly to physiological / neural signals."
)
