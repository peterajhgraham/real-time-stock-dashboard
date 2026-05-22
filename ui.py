"""Streamlit rendering helpers.

All Plotly/Streamlit code lives here so the analysis modules stay free of
UI dependencies. Each ``render_*`` function takes a processed DataFrame and
emits a self-contained panel.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats as sst

from config import PALETTE, RETURNS_HIST_BINS
from spectral import PSDResult, STFTResult
from stats import RegimeResult, ReturnsDistribution


# ---------------------------------------------------------------------------
# Price + overlays
# ---------------------------------------------------------------------------
def build_price_figure(
    data: pd.DataFrame,
    chart_type: str,
    overlays: List[str],
    regime: Optional[RegimeResult] = None,
) -> go.Figure:
    """Main price chart with selectable indicator overlays and regime shading."""
    fig = go.Figure()

    if chart_type == "Candlestick":
        fig.add_trace(
            go.Candlestick(
                x=data["Datetime"],
                open=data["Open"],
                high=data["High"],
                low=data["Low"],
                close=data["Close"],
                name="Price",
            )
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=data["Datetime"],
                y=data["Close"],
                mode="lines",
                name="Close",
                line=dict(color=PALETTE.price),
            )
        )

    overlay_map = {
        "SMA": ("SMA", PALETTE.sma),
        "EMA": ("EMA", PALETTE.ema),
        "Bollinger Bands": None,
    }
    for ind in overlays:
        if ind == "Bollinger Bands":
            fig.add_trace(go.Scatter(x=data["Datetime"], y=data["BB_UPPER"],
                                     name="BB upper", line=dict(color=PALETTE.upper_band, dash="dot")))
            fig.add_trace(go.Scatter(x=data["Datetime"], y=data["BB_LOWER"],
                                     name="BB lower", line=dict(color=PALETTE.lower_band, dash="dot"),
                                     fill="tonexty", fillcolor="rgba(148,103,189,0.10)"))
        elif overlay_map.get(ind):
            col, color = overlay_map[ind]
            fig.add_trace(go.Scatter(x=data["Datetime"], y=data[col], name=ind,
                                     line=dict(color=color)))

    if regime is not None:
        _shade_regimes(fig, data, regime)

    fig.update_layout(
        height=560,
        xaxis_title="Time",
        yaxis_title="Price",
        legend=dict(orientation="h", y=1.02, yanchor="bottom"),
        margin=dict(l=40, r=40, t=40, b=40),
    )
    return fig


def _shade_regimes(fig: go.Figure, data: pd.DataFrame, regime: RegimeResult) -> None:
    """Overlay coloured background bands for HMM-decoded regimes."""
    # Regimes are decoded on returns, which are 1 sample shorter than prices.
    ts = data["Datetime"].iloc[1:].reset_index(drop=True)
    if len(ts) != len(regime.states):
        return
    rank = {s: i for i, s in enumerate(regime.ordering)}  # bear=0 ... bull=N-1
    palette = PALETTE.regime_colors
    states = regime.states
    start = 0
    for i in range(1, len(states) + 1):
        if i == len(states) or states[i] != states[start]:
            color = palette[rank[states[start]] % len(palette)]
            fig.add_vrect(
                x0=ts.iloc[start], x1=ts.iloc[i - 1],
                fillcolor=color, opacity=0.08, layer="below", line_width=0,
            )
            start = i


# ---------------------------------------------------------------------------
# Sub-panels for the new indicators
# ---------------------------------------------------------------------------
def render_macd_panel(data: pd.DataFrame) -> None:
    """MACD bandpass: line, signal, histogram."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data["Datetime"], y=data["MACD"],
                             name="MACD", line=dict(color=PALETTE.macd)))
    fig.add_trace(go.Scatter(x=data["Datetime"], y=data["MACD_SIGNAL"],
                             name="Signal", line=dict(color=PALETTE.signal)))
    fig.add_trace(go.Bar(x=data["Datetime"], y=data["MACD_HIST"],
                         name="Histogram", marker_color=PALETTE.histogram, opacity=0.5))
    fig.update_layout(height=260, margin=dict(l=40, r=40, t=30, b=30),
                      title="MACD - bandpass filter (EMA_fast - EMA_slow)")
    st.plotly_chart(fig, use_container_width=True)


def render_volatility_panel(data: pd.DataFrame) -> None:
    """ATR + rolling z-score side-by-side."""
    col1, col2 = st.columns(2)
    with col1:
        fig = px.line(data, x="Datetime", y="ATR",
                      title="Average True Range - rolling volatility envelope")
        fig.update_layout(height=240, margin=dict(l=40, r=40, t=40, b=30))
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.line(data, x="Datetime", y="ZSCORE",
                      title="Rolling z-score - local standardisation")
        fig.add_hline(y=0, line_dash="dot", line_color="grey")
        fig.add_hline(y=2, line_dash="dot", line_color="red")
        fig.add_hline(y=-2, line_dash="dot", line_color="red")
        fig.update_layout(height=240, margin=dict(l=40, r=40, t=40, b=30))
        st.plotly_chart(fig, use_container_width=True)


def render_obv_panel(data: pd.DataFrame) -> None:
    if "OBV" not in data.columns:
        return
    fig = px.line(data, x="Datetime", y="OBV",
                  title="On-Balance Volume - sign-modulated cumulative integrator")
    fig.update_layout(height=240, margin=dict(l=40, r=40, t=40, b=30))
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Spectral panels
# ---------------------------------------------------------------------------
def render_psd_panel(psd: PSDResult) -> None:
    if psd.freqs.size == 0:
        st.info("Not enough samples for a PSD estimate.")
        return
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=psd.freqs, y=psd.power, mode="lines",
                             name="PSD", line=dict(color=PALETTE.price)))
    if psd.peak_freqs.size:
        fig.add_trace(go.Scatter(
            x=psd.peak_freqs, y=psd.peak_powers, mode="markers+text",
            text=[f"{f:.3f}" for f in psd.peak_freqs],
            textposition="top center",
            marker=dict(color="red", size=9),
            name="Dominant peaks",
        ))
    fig.update_layout(
        title="Welch PSD - dominant frequency content (same operator as EEG band detection)",
        xaxis_title="Frequency (cycles / sample)",
        yaxis_title="Power",
        yaxis_type="log",
        height=320, margin=dict(l=40, r=40, t=40, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_spectrogram_panel(stft: STFTResult) -> None:
    if stft.power_db.size == 0:
        st.info("Not enough samples for a spectrogram.")
        return
    fig = go.Figure(data=go.Heatmap(
        x=stft.times,
        y=stft.freqs,
        z=stft.power_db,
        colorscale="Viridis",
        colorbar=dict(title="dB"),
    ))
    fig.update_layout(
        title="STFT spectrogram - time-resolved spectral content",
        xaxis_title="Time (samples)",
        yaxis_title="Frequency (cycles / sample)",
        height=340, margin=dict(l=40, r=40, t=40, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Statistical panels
# ---------------------------------------------------------------------------
def render_autocorrelation_panel(ac: pd.Series) -> None:
    if ac.empty:
        st.info("Not enough samples for autocorrelation.")
        return
    fig = go.Figure()
    fig.add_trace(go.Bar(x=ac.index, y=ac.values, marker_color=PALETTE.price,
                         name="autocorr"))
    fig.add_hline(y=0, line_color="grey")
    fig.update_layout(
        title="Returns autocorrelation (lags 1..K) - serial dependence",
        xaxis_title="Lag", yaxis_title="rho",
        height=280, margin=dict(l=40, r=40, t=40, b=30),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_returns_distribution_panel(rd: ReturnsDistribution) -> None:
    if rd.returns.size == 0:
        st.info("Not enough samples for a return distribution.")
        return
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=rd.returns, nbinsx=RETURNS_HIST_BINS, histnorm="probability density",
        name="returns", marker_color=PALETTE.price, opacity=0.7,
    ))
    xs = np.linspace(rd.returns.min(), rd.returns.max(), 200)
    pdf = sst.norm.pdf(xs, rd.mu, rd.sigma) if rd.sigma > 0 else np.zeros_like(xs)
    fig.add_trace(go.Scatter(x=xs, y=pdf, mode="lines",
                             line=dict(color="red"), name="Normal fit"))
    fig.update_layout(
        title=(f"Log-returns: mu={rd.mu:.4g}, sigma={rd.sigma:.4g}, "
               f"skew={rd.skew:.3g}, kurt={rd.kurt:.3g}  |  "
               f"KS={rd.ks_stat:.3f} (p={rd.ks_pvalue:.3g})"),
        xaxis_title="log-return", yaxis_title="density",
        height=300, margin=dict(l=40, r=40, t=50, b=30),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_sharpe_panel(sharpe: pd.Series, data: pd.DataFrame) -> None:
    df = pd.DataFrame({"Datetime": data["Datetime"], "Sharpe": sharpe})
    fig = px.line(df, x="Datetime", y="Sharpe",
                  title="Rolling annualised Sharpe ratio")
    fig.add_hline(y=0, line_dash="dot", line_color="grey")
    fig.update_layout(height=260, margin=dict(l=40, r=40, t=40, b=30))
    st.plotly_chart(fig, use_container_width=True)


def render_drawdown_panel(dd: pd.Series, data: pd.DataFrame) -> None:
    df = pd.DataFrame({"Datetime": data["Datetime"], "Drawdown": dd})
    fig = px.area(df, x="Datetime", y="Drawdown",
                  title="Drawdown from running maximum")
    fig.update_traces(line=dict(color="#d62728"), fillcolor="rgba(214,39,40,0.25)")
    fig.update_layout(height=260, margin=dict(l=40, r=40, t=40, b=30))
    st.plotly_chart(fig, use_container_width=True)


def render_regime_summary(regime: RegimeResult) -> None:
    rows = []
    for rank, s in enumerate(regime.ordering):
        label = ["Bear", "Sideways", "Bull"]
        name = label[rank] if len(regime.ordering) == 3 else f"State {rank}"
        rows.append({
            "Regime": name,
            "State #": int(s),
            "Mean log-return": float(regime.means[s]),
            "Variance": float(regime.variances[s]),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
