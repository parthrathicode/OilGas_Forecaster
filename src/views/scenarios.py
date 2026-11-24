"""Scenario Lab - what-if analysis on top of the risk-aware forecast."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

import ui
from core import get_context
from india_data import estimate_indian_fuel_prices, import_bill_impact, indian_basket_from_brent
from views.common import rgba, today_marker

GLOBAL_DEMAND_MBD = 104.0  # approximate world oil demand, million barrels per day

PRESETS = {
    "OPEC+ cut": {"sc_supply": -1.5, "sc_demand": 0.0, "sc_risk": 0.55},
    "Hormuz crisis": {"sc_supply": -3.0, "sc_demand": 0.0, "sc_risk": 0.95},
    "Recession": {"sc_supply": 0.0, "sc_demand": -3.0, "sc_risk": 0.45},
    "Supply glut": {"sc_supply": 2.0, "sc_demand": -0.5, "sc_risk": 0.2},
}


def signed(value, symbol, decimals=2):
    """+$3.20 / −₹1,250 style formatting."""
    sign = "+" if value >= 0 else "−"
    return f"{sign}{symbol}{abs(value):,.{decimals}f}"


def _apply_preset(name, news_risk, inr_rate):
    if name == "Reset":
        values = {"sc_supply": 0.0, "sc_demand": 0.0, "sc_risk": round(news_risk, 2)}
        st.session_state["sc_inr"] = round(inr_rate, 1)
    else:
        values = PRESETS[name]
    for key, value in values.items():
        st.session_state[key] = value


def render():
    ctx = get_context()
    news_risk = float(ctx.news["aggregate_risk"])
    st.session_state.setdefault("sc_supply", 0.0)
    st.session_state.setdefault("sc_demand", 0.0)
    st.session_state.setdefault("sc_risk", round(news_risk, 2))
    st.session_state.setdefault("sc_inr", round(ctx.inr_rate, 1))
    st.session_state.setdefault("sc_elasticity", 0.25)
    st.session_state.setdefault("sc_speed", 7)

    ui.page_header(
        "Forecasting",
        "Scenario lab",
        "Stress-test the 30-day outlook. Supply and demand shocks are layered on the risk-aware LSTM using a "
        "transparent price-elasticity model; the risk input feeds the network directly.",
    )

    controls, results = st.columns([1, 2.1], gap="medium")
    with controls:
        with ui.card("Shock inputs"):
            ui.html_block('<div class="gc-subtle" style="margin-bottom:6px">Presets</div>')
            p1, p2 = st.columns(2)
            for i, name in enumerate(list(PRESETS) + ["Reset"]):
                (p1 if i % 2 == 0 else p2).button(
                    name, key=f"preset_{name}", use_container_width=True,
                    on_click=_apply_preset, args=(name, news_risk, ctx.inr_rate),
                )
            st.slider("Supply change (million bbl/day)", -4.0, 4.0, step=0.25, key="sc_supply",
                      help="Negative = supply lost (OPEC+ cut, outage, blockade). Positive = extra supply.")
            st.slider("Demand shock (% of world demand)", -5.0, 5.0, step=0.5, key="sc_demand",
                      help="Negative = recession / demand destruction. Positive = demand boom.")
            st.slider("Geopolitical risk input", 0.0, 1.0, step=0.05, key="sc_risk",
                      help=f"Feeds the risk-aware LSTM. Today's news risk is {news_risk:.2f}.")
            st.slider("USD / INR", 75.0, 110.0, step=0.5, key="sc_inr")
            with st.expander("Model assumptions"):
                st.slider("Combined price elasticity", 0.10, 0.60, step=0.05, key="sc_elasticity",
                          help="Short-run supply + demand elasticity. 0.25 means a 1% supply loss lifts "
                               "prices by about 4%.")
                st.slider("Days for the market to price it in", 2, 20, key="sc_speed")

    supply = st.session_state.sc_supply
    demand = st.session_state.sc_demand
    risk = st.session_state.sc_risk
    fx = st.session_state.sc_inr
    elasticity = st.session_state.sc_elasticity
    speed = st.session_state.sc_speed

    base_q = ctx.hybrid.forecast_quantiles(ctx.prices, news_risk)
    scen_q = ctx.hybrid.forecast_quantiles(ctx.prices, risk)

    supply_pct = -(supply / GLOBAL_DEMAND_MBD * 100) / elasticity
    demand_pct = demand / elasticity
    total_pct = supply_pct + demand_pct
    t = np.arange(1, len(base_q["median"]) + 1)
    ramp = 1 - np.exp(-t / speed)
    factor = 1 + (total_pct / 100) * ramp

    baseline = base_q["median"]
    scenario = scen_q["median"] * factor
    lower, upper = scen_q["lower"] * factor, scen_q["upper"] * factor
    risk_effect = scen_q["median"][-1] - base_q["median"][-1]
    delta = scenario[-1] - baseline[-1]

    brent_scenario = ctx.brent * scenario[-1] / ctx.spot
    fuel_now = estimate_indian_fuel_prices(ctx.india["indian_basket_usd"], ctx.inr_rate)
    fuel_scen = estimate_indian_fuel_prices(indian_basket_from_brent(brent_scenario), fx)
    petrol_delta = fuel_scen["estimated_petrol_inr"] - fuel_now["estimated_petrol_inr"]
    bill = import_bill_impact(brent_scenario - ctx.brent, fx)

    with results:
        ui.kpi_grid([
            ui.kpi("Scenario · day 30", ui.usd(scenario[-1]), "/bbl", (scenario[-1] / ctx.spot - 1) * 100,
                   sub="vs spot", accent=ui.ACCENT),
            ui.kpi("vs baseline", signed(delta, "$"), "/bbl", (scenario[-1] / baseline[-1] - 1) * 100,
                   sub=f"baseline {ui.usd(baseline[-1])}"),
            ui.kpi("Petrol estimate", ui.inr(fuel_scen["estimated_petrol_inr"]), "/L", None,
                   sub=f"{signed(petrol_delta, '₹')}/L vs today", accent=ui.TEAL),
            ui.kpi("India import bill", signed(bill["usd_billion"], "$", 1) + "bn", "/yr", None,
                   sub=f"{signed(bill['inr_crore'], '₹', 0)} crore a year"),
        ])
        with ui.card("Baseline vs scenario · next 30 trading days"):
            history = ctx.df["Price"].iloc[-63:]
            x = [history.index[-1]] + list(ctx.forecast_dates)
            anchor = ctx.spot
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=history.index, y=history.values, name="WTI history", mode="lines",
                                     line=dict(color=ui.BLUE, width=2),
                                     hovertemplate="$%{y:.2f}<extra>WTI</extra>"))
            fig.add_trace(ui.band_trace(x, [anchor] + list(lower), [anchor] + list(upper),
                                        rgba(ui.ACCENT, 0.14), "Scenario 80% range"))
            fig.add_trace(go.Scatter(x=x, y=[anchor] + list(baseline), name="Baseline (today's news risk)",
                                     mode="lines", line=dict(color=ui.SLATE, width=2, dash="dash"),
                                     hovertemplate="$%{y:.2f}<extra>Baseline</extra>"))
            fig.add_trace(go.Scatter(x=x, y=[anchor] + list(scenario), name="Scenario", mode="lines",
                                     line=dict(color=ui.ACCENT, width=2.8),
                                     hovertemplate="$%{y:.2f}<extra>Scenario</extra>"))
            today_marker(fig, history.index[-1])
            ui.style_fig(fig, height=360, yprefix="$")
            ui.chart(fig)

        a, b = st.columns(2, gap="medium")
        with a:
            with ui.card("Where the change comes from", "Contribution to the day-30 price vs baseline"):
                ui.stat_list([
                    ("Supply shock", f"{supply_pct:+.1f}%  ({supply:+.2f} mb/d)"),
                    ("Demand shock", f"{demand_pct:+.1f}%  ({demand:+.1f}% demand)"),
                    ("Risk input (LSTM)", f"{risk_effect:+.2f} $/bbl  (risk {risk:.2f})"),
                    ("Total vs baseline", f"{delta:+.2f} $/bbl"),
                ])
        with b:
            with ui.card("India pass-through", f"At ₹{fx:.1f} per USD"):
                ui.stat_list([
                    ("Indian basket (scenario)", ui.usd(indian_basket_from_brent(brent_scenario))),
                    ("Petrol estimate", f"{ui.inr(fuel_scen['estimated_petrol_inr'])}/L"),
                    ("Diesel estimate", f"{ui.inr(fuel_scen['estimated_diesel_inr'])}/L"),
                    ("Crude cost per litre", f"{ui.inr(fuel_scen['crude_inr_per_litre'])}"),
                ])
        ui.callout(
            "<b>How this works.</b> Price impact ≈ (shock as % of world demand) ÷ elasticity, phased in "
            "exponentially. A 1 mb/d supply loss is ~1% of world demand, so with elasticity 0.25 it adds "
            "roughly 4% to prices. The risk input changes the LSTM's own forecast and its learned range."
        )
