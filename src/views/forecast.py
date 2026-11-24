"""Short-term (30 trading day) forecast page."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ui
from core import get_context
from views.common import data_pills, forecast_figure, range_picker, rgba

MODELS = ["Price-only LSTM", "Risk-aware LSTM"]
RISK_SOURCES = ["News risk", "Market risk (OVX)", "Custom"]


def render():
    ctx = get_context()
    ui.page_header(
        "Forecasting",
        "Short-term forecast",
        "30-trading-day WTI forecasts from two LSTM networks. The risk-aware model reads a market-stress "
        "channel that you can drive with live news risk, options-market risk, or your own scenario.",
        pills=data_pills(ctx),
    )

    with ui.card():
        c1, c2, c3 = st.columns([1.15, 1.6, 1], gap="medium")
        model = c1.segmented_control("Model", MODELS, default=MODELS[1], key="fc_model") or MODELS[1]
        risk_aware = model == MODELS[1]
        source = c2.segmented_control("Risk input", RISK_SOURCES, default=RISK_SOURCES[0], key="fc_source",
                                      disabled=not risk_aware) or RISK_SOURCES[0]
        if source == "News risk":
            risk = float(ctx.news["aggregate_risk"])
            c3.slider("Risk level", 0.0, 1.0, round(risk, 2), 0.01, disabled=True, key="fc_risk_news",
                      help="Average risk score of today's scored headlines")
        elif source == "Market risk (OVX)":
            risk = float(ctx.market_risk)
            c3.slider("Risk level", 0.0, 1.0, round(risk, 2), 0.01, disabled=True, key="fc_risk_mkt",
                      help="Percentile of the CBOE Oil Volatility Index over the last 3 years")
        else:
            risk = c3.slider("Risk level", 0.0, 1.0, 0.5, 0.05, key="fc_risk_custom", disabled=not risk_aware,
                             help="0 = calm markets, 1 = extreme stress")

    if risk_aware:
        q = ctx.hybrid.forecast_quantiles(ctx.prices, risk)
        path, lower, upper = q["median"], q["lower"], q["upper"]
        color, band_name = ui.TEAL, "Learned 80% range"
    else:
        path, lower, upper = ctx.std_path, ctx.std_lower, ctx.std_upper
        color, band_name = ui.ACCENT, "80% interval"

    spot = ctx.spot
    chg = lambda v: (v / spot - 1) * 100  # noqa: E731
    ui.kpi_grid([
        ui.kpi("Spot (WTI)", ui.usd(spot), "/bbl", ctx.spot_change, sub="day change", accent=ui.BLUE),
        ui.kpi("Day 7", ui.usd(path[6]), "", chg(path[6]), sub="vs spot"),
        ui.kpi("Day 14", ui.usd(path[13]), "", chg(path[13]), sub="vs spot"),
        ui.kpi("Day 30", ui.usd(path[-1]), "", chg(path[-1]), sub="vs spot", accent=color),
        ui.kpi("80% range · day 30", f"{ui.usd(lower[-1], 0)}–{ui.usd(upper[-1], 0)}", "",
               sub=f"width {ui.usd(upper[-1] - lower[-1], 0)}"),
    ])

    with ui.card(f"{model} · 30-day path",
                 f"Risk input {risk:.2f}" if risk_aware else "Interval from walk-forward validation errors"):
        days = range_picker("fc_range", default="3M", options=("3M", "6M", "1Y"))
        history = ctx.df["Price"].iloc[-days:]
        fig = forecast_figure(history, ctx.forecast_dates, path, lower, upper, color=color,
                              name=model, band_name=band_name, height=400)
        ui.chart(fig)

    left, right = st.columns([1, 1], gap="medium")
    with left:
        with ui.card("Risk sensitivity", "What the risk-aware model learned from 15 years of OVX history"):
            levels = np.round(np.linspace(0, 1, 11), 2)
            sens = [ctx.hybrid.forecast_quantiles(ctx.prices, lv) for lv in levels]
            med = [s["median"][-1] for s in sens]
            lo = [s["lower"][-1] for s in sens]
            hi = [s["upper"][-1] for s in sens]
            fig = go.Figure()
            fig.add_trace(ui.band_trace(levels, lo, hi, rgba(ui.TEAL, 0.18), "80% range (day 30)"))
            fig.add_trace(go.Scatter(x=levels, y=med, name="Median (day 30)", mode="lines+markers",
                                     line=dict(color=ui.TEAL, width=2.4), marker=dict(size=6),
                                     hovertemplate="risk %{x:.1f}: $%{y:.2f}<extra></extra>"))
            if risk_aware:
                fig.add_vline(x=risk, line=dict(color=ui.ACCENT, dash="dot", width=1.4))
            ui.style_fig(fig, height=300, yprefix="$", hovermode="closest")
            fig.update_xaxes(title=dict(text="Risk input", font=dict(color=ui.SUBTLE, size=11)))
            ui.chart(fig)
            width_lo, width_hi = hi[0] - lo[0], hi[-1] - lo[-1]
            ui.callout(
                f"Moving risk from 0 to 1 widens the day-30 range from <b>{ui.usd(width_lo, 0)}</b> to "
                f"<b>{ui.usd(width_hi, 0)}</b> and shifts the median by <b>{ui.usd(med[-1] - med[0])}</b>. "
                "Stress mostly predicts <i>how far</i> prices may move, more than <i>which way</i>."
            )

    with right:
        with ui.card("Forecast table", "Trading-day path with rupee equivalent"):
            table = pd.DataFrame({
                "Date": ctx.forecast_dates.strftime("%d %b %Y"),
                "Forecast ($)": np.round(path, 2),
                "Low ($)": np.round(lower, 2),
                "High ($)": np.round(upper, 2),
                "Forecast (₹/bbl)": np.round(path * ctx.inr_rate, 0),
            })
            st.dataframe(table, hide_index=True, use_container_width=True, height=300)
            st.download_button("Download CSV", table.to_csv(index=False).encode(), "wti_30day_forecast.csv",
                               "text/csv", icon=":material/download:")
