"""
Market data layer
=================
Downloads daily market history from Yahoo Finance:

    CL=F    WTI crude oil futures (front month)        USD / barrel
    BZ=F    Brent crude oil futures (front month)      USD / barrel
    ^OVX    CBOE Crude Oil Volatility Index            implied volatility, %
    INR=X   US dollar / Indian rupee spot rate         INR per USD

Every series falls back to a reproducible synthetic substitute when the
network is unavailable. The returned ``live`` flags record which series are
real, so the interface can label offline data honestly.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:  # pragma: no cover - handled by the fallback path
    yf = None

HISTORY_START = "2000-01-01"   # long-term model uses everything; the LSTMs train from 2010

SYMBOLS = {
    "wti": "CL=F",
    "brent": "BZ=F",
    "ovx": "^OVX",
    "usdinr": "INR=X",
}

# Rolling window (trading days) used to turn OVX into a 0-1 risk percentile.
RISK_WINDOW = 756  # ~3 years


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------
def _fetch_close(symbol: str, start: str = HISTORY_START) -> pd.Series:
    """Daily closing prices for ``symbol`` as a tz-naive, de-duplicated Series."""
    if yf is None:
        raise RuntimeError("yfinance is not installed")
    hist = yf.Ticker(symbol).history(start=start, auto_adjust=False, actions=False)
    if hist is None or hist.empty or "Close" not in hist:
        raise ValueError(f"Yahoo Finance returned no data for {symbol}")

    close = hist["Close"].astype(float)
    index = pd.to_datetime(close.index)
    if index.tz is not None:
        index = index.tz_localize(None)
    close.index = index.normalize()
    close = close[~close.index.duplicated(keep="last")].sort_index().dropna()
    if len(close) < 50:
        raise ValueError(f"Too little history for {symbol} ({len(close)} rows)")
    return close


def clean_wti(series: pd.Series) -> pd.Series:
    """
    Removes non-positive settlements. On 20-Apr-2020 the expiring May WTI
    contract settled at -$37.63, a one-day delivery anomaly that would
    otherwise dominate every relative-return calculation downstream.
    """
    return series.where(series > 0).ffill().dropna()


# ---------------------------------------------------------------------------
# Offline fallbacks
# ---------------------------------------------------------------------------
def _business_index(start: str = HISTORY_START) -> pd.DatetimeIndex:
    return pd.bdate_range(start=start, end=pd.Timestamp.today().normalize())


def _synthetic_mean_reverting(index, start, mean, kappa, sigma, seed) -> pd.Series:
    """Log Ornstein-Uhlenbeck path: realistic-looking, mean-reverting prices."""
    rng = np.random.default_rng(seed)
    log_x = np.empty(len(index))
    log_x[0] = np.log(start)
    log_mean = np.log(mean)
    shocks = rng.normal(0.0, sigma, len(index))
    for t in range(1, len(index)):
        log_x[t] = log_x[t - 1] + kappa * (log_mean - log_x[t - 1]) + shocks[t]
    return pd.Series(np.exp(log_x), index=index)


def _fallback_wti() -> pd.Series:
    index = _business_index()
    return _synthetic_mean_reverting(index, start=80.0, mean=72.0, kappa=0.004, sigma=0.021, seed=7)


def _fallback_brent(wti: pd.Series) -> pd.Series:
    rng = np.random.default_rng(11)
    spread = 4.0 + np.cumsum(rng.normal(0, 0.05, len(wti))).clip(-3, 3)
    return (wti + spread).rename(None)


def _fallback_ovx(wti: pd.Series) -> pd.Series:
    """Realised 20-day volatility of WTI, annualised, as a stand-in for OVX."""
    realised = np.log(wti).diff().rolling(20, min_periods=5).std() * np.sqrt(252) * 100
    return realised.bfill().clip(lower=10)


def _fallback_usdinr() -> pd.Series:
    index = _business_index()
    rng = np.random.default_rng(23)
    trend = np.linspace(np.log(45.0), np.log(88.0), len(index))
    noise = np.cumsum(rng.normal(0, 0.002, len(index)))
    noise -= np.linspace(0, noise[-1], len(index))  # pin both ends to the trend
    return pd.Series(np.exp(trend + noise), index=index)


# ---------------------------------------------------------------------------
# Derived signals
# ---------------------------------------------------------------------------
def market_risk_index(ovx: pd.Series, index: pd.DatetimeIndex, window: int = RISK_WINDOW) -> pd.Series:
    """
    Converts OVX into a 0-1 market-stress score: the percentile of today's
    implied volatility within the trailing ``window`` days. Uses only past
    data, so it is safe to use as a model input without look-ahead bias.
    """
    aligned = ovx.reindex(index.union(ovx.index)).sort_index().ffill().reindex(index)
    pct = aligned.rolling(window, min_periods=60).rank(pct=True)
    return pct.fillna(0.5).clip(0.0, 1.0)


def _last_change(series: pd.Series) -> tuple[float, float]:
    """(latest value, % change versus the previous observation)."""
    latest = float(series.iloc[-1])
    previous = float(series.iloc[-2]) if len(series) > 1 else latest
    change = (latest / previous - 1.0) * 100 if previous else 0.0
    return latest, change


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def load_market_data() -> dict:
    """
    Downloads (or synthesises) every series the platform needs.

    Returns a dict with:
        wti         DataFrame, index=date, column 'Price' (USD/bbl)
        brent       Series (USD/bbl)
        ovx         Series aligned to the WTI index
        usdinr      Series (INR per USD)
        risk_index  Series aligned to the WTI index, 0-1 market stress
        live        dict[str, bool]  - True when the series is real data
        quotes      dict[str, (latest, pct_change)]
        fetched_at  datetime of the download
    """
    live: dict[str, bool] = {}

    try:
        wti = clean_wti(_fetch_close(SYMBOLS["wti"]))
        live["wti"] = True
    except Exception as exc:  # network, rate-limit, parsing...
        print(f"[market_data] WTI download failed ({exc}); using offline sample data.")
        wti = _fallback_wti()
        live["wti"] = False

    try:
        brent = _fetch_close(SYMBOLS["brent"])
        live["brent"] = True
    except Exception as exc:
        print(f"[market_data] Brent download failed ({exc}); deriving from WTI.")
        brent = _fallback_brent(wti)
        live["brent"] = False

    try:
        ovx = _fetch_close(SYMBOLS["ovx"])
        live["ovx"] = True
    except Exception as exc:
        print(f"[market_data] OVX download failed ({exc}); using realised volatility.")
        ovx = _fallback_ovx(wti)
        live["ovx"] = False

    try:
        usdinr = _fetch_close(SYMBOLS["usdinr"])
        # Yahoo occasionally prints obviously broken INR ticks; drop them.
        usdinr = usdinr[(usdinr > 30) & (usdinr < 200)]
        live["usdinr"] = True
    except Exception as exc:
        print(f"[market_data] USD/INR download failed ({exc}); using an estimate.")
        usdinr = _fallback_usdinr()
        live["usdinr"] = False

    wti_df = wti.to_frame("Price")
    wti_df.index.name = "Date"
    ovx_aligned = ovx.reindex(wti_df.index.union(ovx.index)).sort_index().ffill().reindex(wti_df.index).bfill()

    return {
        "wti": wti_df,
        "brent": brent,
        "ovx": ovx_aligned,
        "usdinr": usdinr,
        "risk_index": market_risk_index(ovx, wti_df.index),
        "live": live,
        "quotes": {
            "wti": _last_change(wti),
            "brent": _last_change(brent),
            "ovx": _last_change(ovx),
            "usdinr": _last_change(usdinr),
        },
        "fetched_at": dt.datetime.now(),
    }


if __name__ == "__main__":
    data = load_market_data()
    print("Live flags:", data["live"])
    print(data["wti"].tail())
    for name, (value, change) in data["quotes"].items():
        print(f"{name:7s} {value:10.2f}  {change:+.2f}%")
    print("Latest market risk index:", round(float(data["risk_index"].iloc[-1]), 3))
