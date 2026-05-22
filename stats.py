"""Statistical diagnostics for streaming time-series.

Autocorrelation, returns distributions, rolling Sharpe, drawdown, and an
optional Hidden Markov Model regime decoder. Each routine is written so
that it applies equally well to log-returns of an equity series, the
amplitude envelope of an LFP recording, or any other broadly-stationary
signal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats as sst

from config import (
    ANNUALISATION_FACTOR,
    AUTOCORR_MAX_LAG,
    HMM_N_ITER,
    HMM_N_STATES,
    HMM_RANDOM_SEED,
    SHARPE_WINDOW,
)


def log_returns(close: pd.Series) -> pd.Series:
    """Log-returns r_t = log(p_t) - log(p_{t-1}).

    Log-returns are additive across time and approximately stationary -
    the same reason neuroscientists work with log-power rather than raw
    power in EEG band analysis.
    """
    return np.log(close).diff()


def autocorrelation(x: pd.Series, max_lag: int = AUTOCORR_MAX_LAG) -> pd.Series:
    """Sample autocorrelation at lags 1..max_lag.

    rho(k) = Cov(x_t, x_{t-k}) / Var(x_t)

    Detects serial dependence in the signal. Non-zero autocorrelation in
    returns indicates predictable structure; in spike trains the same
    statistic reveals refractoriness and rhythmicity.
    """
    arr = x.dropna().to_numpy()
    if arr.size < max_lag + 2:
        return pd.Series(dtype=float)
    arr = arr - arr.mean()
    var = np.dot(arr, arr)
    if var == 0:
        return pd.Series(np.zeros(max_lag), index=range(1, max_lag + 1))
    ac = np.array([np.dot(arr[:-k], arr[k:]) / var for k in range(1, max_lag + 1)])
    return pd.Series(ac, index=pd.Index(range(1, max_lag + 1), name="lag"))


@dataclass
class ReturnsDistribution:
    returns: np.ndarray
    mu: float
    sigma: float
    skew: float
    kurt: float
    ks_stat: float               # Kolmogorov-Smirnov vs. fitted normal
    ks_pvalue: float


def returns_distribution(close: pd.Series) -> ReturnsDistribution:
    """Empirical return distribution with Gaussian goodness-of-fit.

    Fits a normal by maximum likelihood, then runs a Kolmogorov-Smirnov
    test against that fit. Equity returns are famously heavy-tailed; the
    same diagnostic is used to test whether trial-to-trial neural firing
    rates are well-modelled by a Gaussian envelope.
    """
    r = log_returns(close).dropna().to_numpy()
    if r.size < 8:
        return ReturnsDistribution(r, 0.0, 0.0, 0.0, 0.0, float("nan"), float("nan"))
    mu, sigma = float(r.mean()), float(r.std(ddof=1))
    ks_stat, ks_p = sst.kstest(r, "norm", args=(mu, sigma))
    return ReturnsDistribution(
        returns=r,
        mu=mu,
        sigma=sigma,
        skew=float(sst.skew(r)),
        kurt=float(sst.kurtosis(r)),
        ks_stat=float(ks_stat),
        ks_pvalue=float(ks_p),
    )


def rolling_sharpe(
    close: pd.Series,
    window: int = SHARPE_WINDOW,
    annualisation: float = ANNUALISATION_FACTOR,
) -> pd.Series:
    """Rolling annualised Sharpe ratio of log-returns.

    SR_t = sqrt(A) * mean(r_{t-W:t}) / std(r_{t-W:t})

    A signal-to-noise ratio applied to financial returns. Conceptually
    identical to d-prime in psychophysics or SNR estimation in neural
    decoding - mean signal divided by its variability, scaled to a unit
    of interest.
    """
    r = log_returns(close)
    mean = r.rolling(window=window, min_periods=window).mean()
    sd = r.rolling(window=window, min_periods=window).std(ddof=0)
    return np.sqrt(annualisation) * (mean / sd)


def drawdown(close: pd.Series) -> pd.Series:
    """Normalised drawdown from the running maximum.

    dd_t = close_t / max_{s <= t}(close_s) - 1

    Tracks how far the signal sits below its prior peak. The same
    construction surfaces fatigue effects in behavioural time-series
    (e.g. accuracy relative to a session-best baseline).
    """
    running_max = close.cummax()
    return close / running_max - 1.0


# ---------------------------------------------------------------------------
# Optional HMM regime decoder
# ---------------------------------------------------------------------------
@dataclass
class RegimeResult:
    states: np.ndarray           # decoded state per time step (aligned to returns)
    means: np.ndarray            # state means
    variances: np.ndarray        # state variances
    ordering: np.ndarray         # state indices sorted by mean (bear -> bull)


def fit_regime_hmm(
    close: pd.Series,
    n_states: int = HMM_N_STATES,
    n_iter: int = HMM_N_ITER,
    seed: int = HMM_RANDOM_SEED,
) -> Optional[RegimeResult]:
    """Two- or three-state Gaussian HMM on log-returns.

    Each hidden state has its own Gaussian emission; Viterbi decoding
    assigns every time step to the most likely state. By mean-return
    ordering these correspond to bear / sideways / bull regimes.

    Structurally identical to the behavioural-state decoding used in
    ``neural-representation-explorer`` - same generative model, different
    domain. Returns ``None`` if ``hmmlearn`` is not installed or the
    series is too short to fit.
    """
    try:
        from hmmlearn.hmm import GaussianHMM
    except Exception:
        return None

    r = log_returns(close).dropna().to_numpy().reshape(-1, 1)
    if r.shape[0] < max(50, 10 * n_states):
        return None

    model = GaussianHMM(
        n_components=n_states,
        covariance_type="diag",
        n_iter=n_iter,
        random_state=seed,
    )
    try:
        model.fit(r)
        states = model.predict(r)
    except Exception:
        return None

    means = model.means_.flatten()
    variances = model.covars_.reshape(-1)
    ordering = np.argsort(means)        # bear (low mean) ... bull (high mean)
    return RegimeResult(states=states, means=means, variances=variances, ordering=ordering)
