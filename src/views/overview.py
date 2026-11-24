"""Market overview - the landing page."""

from __future__ import annotations

import streamlit as st

import nav
import ui
from core import get_context
from views.common import data_pills, forecast_figure, range_picker


def render():
    ctx = get_context()
    gauge = ctx.news_gauge
    d30 = (ctx.std_path[-1] / ctx.spot - 1) * 100

    ui.page_header(
        "Markets",
        "Market overview",
        "Live crude prices, AI forecasts and geopolitical risk in one view.",
        pills=data_pills(ctx),
    )

    ui.kpi_grid([
        ui.kpi("WTI crude", ui.usd(ctx.spot), "/bbl", ctx.spot_change, sub="day change", accent=ui.BLUE),
        ui.kpi("Brent crude", ui.usd(ctx.brent), "/bbl", ctx.brent_change, sub="day change"),
        ui.kpi("30-day forecast", ui.usd(ctx.std_path[-1]), "/bbl", d30, sub="LSTM vs today", accent=ui.ACCENT),
        ui.kpi("USD / INR", f"₹{ctx.inr_rate:.2f}", "", ctx.inr_change, sub="day change", invert=True),
        ui.kpi("News risk index", f"{ctx.news['aggregate_risk']:.2f}", "", tag=(gauge["level"], gauge["color"]),
               sub=f"{ctx.news['total_analyzed']} headlines", accent=gauge["color"]),
    ])

    left, right = st.columns([2.1, 1], gap="medium")
    with left:
        with ui.card("WTI crude · price & 30-day forecast",
                     "Price-only LSTM with an 80% interval from out-of-sample errors"):
            days = range_picker("overview_range", default="6M")
            history = ctx.df["Price"].iloc[-days:]
            fig = forecast_figure(history, ctx.forecast_dates, ctx.std_path, ctx.std_lower, ctx.std_upper,
                                  name="LSTM forecast", height=360)
            ui.chart(fig)

    with right:
        with ui.card("Forecast snapshot", f"From {ctx.last_date:%d %b %Y}, trading days"):
            p = ctx.std_path
            bias = "Bullish" if p[-1] > ctx.spot * 1.005 else ("Bearish" if p[-1] < ctx.spot * 0.995 else "Flat")
            ui.stat_list([
                ("Spot", ui.usd(ctx.spot)),
                ("Day 7", f"{ui.usd(p[6])}  ({ui.pct((p[6] / ctx.spot - 1) * 100)})"),
                ("Day 14", f"{ui.usd(p[13])}  ({ui.pct((p[13] / ctx.spot - 1) * 100)})"),
                ("Day 30", f"{ui.usd(p[-1])}  ({ui.pct(d30)})"),
                ("80% range, day 30", f"{ui.usd(ctx.std_lower[-1], 0)} – {ui.usd(ctx.std_upper[-1], 0)}"),
                ("Model bias", bias),
            ])
        with ui.card("Market stress"):
            ui.stat_list([
                ("OVX (oil volatility index)", f"{ctx.ovx:.1f}"),
                ("OVX 3-year percentile", f"{ctx.market_risk * 100:.0f}th"),
                ("News risk", f"{ctx.news['aggregate_risk']:.2f} · {gauge['level']}"),
            ])

    left, right = st.columns([1.6, 1], gap="medium")
    with left:
        with ui.card("Top headlines", "Scored by FinBERT + energy-domain signals"):
            top = sorted(ctx.news["headlines"], key=lambda h: h["published"], reverse=True)[:4]
            ui.html_block("".join(ui.news_card(h) for h in top))
            st.page_link(nav.get("news"), label="Open the full news feed", icon=":material/arrow_forward:")

    with right:
        with ui.card("India snapshot", "Cost-based Delhi retail estimates"):
            ind = ctx.india
            pt = ind["passthrough"]
            ui.stat_list([
                ("Indian basket (proxy)", f"{ui.usd(ind['indian_basket_usd'])} · {ui.inr(ind['indian_basket_inr'], 0)}"),
                ("Petrol estimate", f"{ui.inr(ind['petrol_estimate_inr'])}/L"),
                ("Diesel estimate", f"{ui.inr(ind['diesel_estimate_inr'])}/L"),
                ("Petrol per $1/bbl", f"{ui.inr(pt['petrol']['per_usd'])}/L"),
                ("Petrol per ₹1 on USD/INR", f"{ui.inr(pt['petrol']['per_inr'])}/L"),
            ])
            st.page_link(nav.get("india"), label="India impact analysis", icon=":material/arrow_forward:")

    st.write("")
    ui.html_block('<div class="gc-card-title" style="margin:6px 0 10px">How the platform works</div>')
    ui.steps([
        ("01 · Signals", "News intelligence",
         "Live Google News headlines scored by FinBERT and an energy-domain lexicon into a 0–1 risk index.",
         ui.ORANGE),
        ("02 · Short term", "LSTM forecasters",
         "Two PyTorch LSTMs read 30 trading days and forecast the next 30 - one price-only, one risk-aware.",
         ui.ACCENT),
        ("03 · Long term", "XGBoost outlook",
         "Gradient-boosted trees on monthly momentum, volatility and mean-reversion features project 36 months.",
         ui.VIOLET),
        ("04 · Impact", "India translation",
         "Forecasts become rupee costs, petrol and diesel estimates and import-bill sensitivities.",
         ui.TEAL),
    ])
