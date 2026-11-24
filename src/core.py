"""
Service layer shared by every page.

Heavy work (downloads, model training, backtests) is cached twice:
* in memory with Streamlit's cache, so page switches and widget changes are
  instant, and
* on disk under ``.model_cache/`` keyed by the latest market date, so models
  are trained once per trading day rather than on every app restart.
"""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import streamlit as st

from backtester import ModelBacktester
from india_data import get_india_market_summary
from longterm_predictor import LongTermForecaster
from market_data import load_market_data
from news_feed import get_risk_gauge_data, score_news_feed
from nlp_scorer import GeopoliticalRiskScorer
from price_predictor import HORIZON, HybridPricePredictorEngine, PricePredictorEngine

MODEL_VERSION = "2.3"
LSTM_START = pd.Timestamp("2010-01-01")
CACHE_DIR = Path(__file__).resolve().parent.parent / ".model_cache"
NEWS_LIMIT = 15


# ---------------------------------------------------------------------------
# Disk cache
# ---------------------------------------------------------------------------
def _disk_cached(name: str, key: str, builder):
    CACHE_DIR.mkdir(exist_ok=True)
    digest = hashlib.sha1(f"{name}|{key}|{MODEL_VERSION}".encode()).hexdigest()[:16]
    path = CACHE_DIR / f"{name}-{digest}.pkl"
    if path.exists():
        try:
            return pickle.loads(path.read_bytes())
        except Exception:
            path.unlink(missing_ok=True)
    result = builder()
    for stale in CACHE_DIR.glob(f"{name}-*.pkl"):
        stale.unlink(missing_ok=True)
    try:
        path.write_bytes(pickle.dumps(result))
    except Exception as exc:  # caching is an optimisation, never fatal
        print(f"[core] could not write cache {path.name}: {exc}")
    return result


def data_key(md: dict) -> str:
    """
    Identifies the training data by its last trading day. Intraday updates to
    today's price don't trigger retraining - forecasts always use the latest
    prices at inference time - but a new trading day does.
    """
    idx = md["wti"].index
    return f"{idx[-1]:%Y-%m-%d}|{len(idx)}|{md['live']['wti']}|{md['live']['ovx']}"


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def get_market_data():
    return load_market_data()


@st.cache_resource(show_spinner=False)
def get_nlp_engine():
    return GeopoliticalRiskScorer()


@st.cache_data(ttl=1800, show_spinner=False)
def get_news(_scorer, limit: int = NEWS_LIMIT):
    return score_news_feed(_scorer, max_headlines=limit)


@st.cache_resource(show_spinner=False)
def get_short_term_models(key: str, _prices, _risk, _dates):
    def build():
        std = PricePredictorEngine().fit(_prices)
        hybrid = HybridPricePredictorEngine().fit(_prices, _risk)
        backtest = ModelBacktester()
        backtest.run_walkforward(_prices, _dates, _risk)
        return {"std": std, "hybrid": hybrid, "backtest": backtest}

    return _disk_cached("short_term", key, build)


@st.cache_resource(show_spinner=False)
def get_long_term_model(key: str, _df):
    return _disk_cached("long_term", key, lambda: LongTermForecaster().train(_df))


@st.cache_resource(show_spinner=False)
def get_crisis_tests(key: str, _prices, _dates, _risk):
    def build():
        bt = ModelBacktester()
        bt.run_crisis_tests(_prices, _dates, _risk)
        return bt.crisis

    return _disk_cached("crisis", key, build)


def refresh_market_data():
    get_market_data.clear()


def refresh_news():
    get_news.clear()


# ---------------------------------------------------------------------------
# Context assembly
# ---------------------------------------------------------------------------
BOOT_STEPS = [
    "Downloading market data",
    "Loading FinBERT language model",
    "Scoring live news",
    "Training LSTM forecasters & walk-forward validation",
    "Training long-term XGBoost model",
]


def get_context(progress=None) -> SimpleNamespace:
    """Everything a page needs, assembled from cached components."""

    def step(i):
        if progress:
            progress(i, BOOT_STEPS[i])

    step(0)
    md = get_market_data()
    key = data_key(md)
    df = md["wti"]
    # The LSTMs train on 2010+ (when the OVX risk channel is fully available);
    # the long-term model uses the full history back to 2000.
    lstm_mask = df.index >= LSTM_START
    prices = df["Price"].to_numpy(dtype=float)[lstm_mask]
    dates = df.index[lstm_mask]
    risk = md["risk_index"].to_numpy(dtype=float)[lstm_mask]

    step(1)
    nlp = get_nlp_engine()
    step(2)
    news = get_news(nlp)
    step(3)
    st_models = get_short_term_models(key, prices, risk, dates)
    step(4)
    lt = get_long_term_model(key, df)

    spot = float(prices[-1])
    last_date = dates[-1]
    forecast_dates = pd.bdate_range(last_date + pd.offsets.BDay(1), periods=HORIZON)

    backtest: ModelBacktester = st_models["backtest"]
    std_path = st_models["std"].forecast(prices)
    iv = backtest.intervals.get("lstm")
    std_lower = std_path * np.exp(iv["lo"]) if iv else None
    std_upper = std_path * np.exp(iv["hi"]) if iv else None

    brent, brent_chg = md["quotes"]["brent"]
    inr_rate, inr_chg = md["quotes"]["usdinr"]
    india = get_india_market_summary(brent, inr_rate, md["live"]["usdinr"], inr_chg)
    lt_forecast = lt.predict_longterm(months=36)
    trend = LongTermForecaster.get_trend_analysis(lt_forecast, spot)

    return SimpleNamespace(
        md=md,
        key=key,
        df=df,
        prices=prices,
        dates=dates,
        risk=risk,
        market_risk=float(risk[-1]),
        spot=spot,
        spot_change=md["quotes"]["wti"][1],
        brent=brent,
        brent_change=brent_chg,
        inr_rate=inr_rate,
        inr_change=inr_chg,
        ovx=md["quotes"]["ovx"][0],
        last_date=last_date,
        live=md["live"],
        all_live=all(md["live"].values()),
        nlp=nlp,
        news=news,
        news_gauge=get_risk_gauge_data(news["aggregate_risk"]),
        std=st_models["std"],
        hybrid=st_models["hybrid"],
        backtest=backtest,
        validation=backtest.headline_metrics(),
        forecast_dates=forecast_dates,
        std_path=std_path,
        std_lower=std_lower,
        std_upper=std_upper,
        longterm=lt,
        lt_forecast=lt_forecast,
        lt_trend=trend,
        india=india,
    )


def chat_context(ctx: SimpleNamespace) -> dict:
    """Snapshot handed to the assistant on every run."""
    band = None
    if ctx.std_lower is not None:
        band = (float(ctx.std_lower[-1]), float(ctx.std_upper[-1]))
    return {
        "wti": ctx.spot,
        "brent": ctx.brent,
        "inr": ctx.inr_rate,
        "basket": ctx.india["indian_basket_usd"],
        "petrol": ctx.india["petrol_estimate_inr"],
        "diesel": ctx.india["diesel_estimate_inr"],
        "passthrough": ctx.india["passthrough"],
        "forecast": ctx.std_path,
        "forecast_band": band,
        "longterm": ctx.lt_trend,
        "news": ctx.news,
        "validation": ctx.validation,
        "market_risk": ctx.market_risk,
    }
