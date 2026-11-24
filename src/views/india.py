"""India impact page - rupee costs, retail fuel build-up and sensitivities."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

import ui
from core import get_context
from india_data import (INDIA_CRUDE_IMPORTS_MBD, estimate_indian_fuel_prices, import_bill_impact)
from views.common import data_pills

COMPONENT_COLORS = {
    "Crude cost": ui.ACCENT,
    "Refining, freight & margin": "#B07A1B",
    "Central excise": ui.VIOLET,
    "Dealer commission": ui.TEAL,
    "State VAT": ui.BLUE,
}


def render():
    ctx = get_context()
    ind = ctx.india
    fuel = ind["fuel"]
    pt = ind["passthrough"]

    ui.page_header(
        "India",
        "India impact",
        "What global crude means at home: the Indian crude basket in rupees, a cost-based build-up of "
        "Delhi petrol and diesel prices, and how sensitive they are to crude and the exchange rate.",
        pills=data_pills(ctx),
    )

    ui.kpi_grid([
        ui.kpi("USD / INR", f"₹{ctx.inr_rate:.2f}", "", ctx.inr_change, sub="day change (▲ = weaker ₹)",
               invert=True, accent=ui.BLUE),
        ui.kpi("Indian basket", ui.usd(ind["indian_basket_usd"]), "/bbl", ctx.brent_change, sub="Brent-linked proxy"),
        ui.kpi("Basket in rupees", ui.inr(ind["indian_basket_inr"], 0), "/bbl", sub="import cost per barrel"),
        ui.kpi("Petrol estimate", ui.inr(ind["petrol_estimate_inr"]), "/L", sub="cost-based, Delhi",
               accent=ui.ACCENT),
        ui.kpi("Diesel estimate", ui.inr(ind["diesel_estimate_inr"]), "/L", sub="cost-based, Delhi",
               accent=ui.TEAL),
    ])

    left, right = st.columns([1.5, 1], gap="medium")
    with left:
        with ui.card("Retail price build-up", "Rupees per litre, from crude to the pump"):
            fig = go.Figure()
            for component, color in COMPONENT_COLORS.items():
                values = [fuel["breakdown"]["petrol"][component], fuel["breakdown"]["diesel"][component]]
                fig.add_trace(go.Bar(
                    y=["Petrol", "Diesel"], x=values, name=component, orientation="h",
                    marker=dict(color=color, line=dict(width=0)),
                    text=[f"₹{v:.1f}" for v in values], textposition="inside", insidetextanchor="middle",
                    textfont=dict(color="#0B0F17", size=11),
                    hovertemplate=component + ": ₹%{x:.2f}/L<extra></extra>",
                ))
            totals = [fuel["estimated_petrol_inr"], fuel["estimated_diesel_inr"]]
            for label, total in zip(["Petrol", "Diesel"], totals):
                fig.add_annotation(x=total, y=label, text=f"<b>₹{total:.2f}</b>", showarrow=False,
                                   xanchor="left", xshift=8, font=dict(color=ui.TEXT, size=13))
            ui.style_fig(fig, height=250, hovermode="closest")
            fig.update_layout(barmode="stack", bargap=0.38)
            fig.update_xaxes(range=[0, max(totals) * 1.14], tickprefix="₹")
            fig.update_yaxes(tickfont=dict(color=ui.TEXT, size=13))
            ui.chart(fig)

    with right:
        with ui.card("Pass-through", "How much the pump price moves"):
            ui.stat_list([
                ("Petrol per $1/bbl crude", f"{ui.inr(pt['petrol']['per_usd'])}/L"),
                ("Diesel per $1/bbl crude", f"{ui.inr(pt['diesel']['per_usd'])}/L"),
                ("Petrol per ₹1 move in USD/INR", f"{ui.inr(pt['petrol']['per_inr'])}/L"),
                ("Diesel per ₹1 move in USD/INR", f"{ui.inr(pt['diesel']['per_inr'])}/L"),
                ("Import bill per $1/bbl", f"${import_bill_impact(1, ctx.inr_rate)['usd_billion']:.2f} bn/yr"),
            ])
            ui.callout(
                "Retail prices in India are administered: oil marketing companies often absorb crude swings "
                "for months. These figures are <b>market-linked estimates</b> of where costs point, not "
                "announced pump prices."
            )

    left, right = st.columns([1, 1], gap="medium")
    with left:
        with ui.card("30-day fuel-cost outlook", "Petrol & diesel estimates along the LSTM crude forecast"):
            ratio = np.concatenate([[1.0], ctx.std_path / ctx.spot])
            basket_path = ind["indian_basket_usd"] * ratio
            est = [estimate_indian_fuel_prices(b, ctx.inr_rate) for b in basket_path]
            x = [ctx.last_date] + list(ctx.forecast_dates)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x, y=[e["estimated_petrol_inr"] for e in est], name="Petrol",
                                     line=dict(color=ui.ACCENT, width=2.6),
                                     hovertemplate="₹%{y:.2f}/L<extra>Petrol</extra>"))
            fig.add_trace(go.Scatter(x=x, y=[e["estimated_diesel_inr"] for e in est], name="Diesel",
                                     line=dict(color=ui.TEAL, width=2.6),
                                     hovertemplate="₹%{y:.2f}/L<extra>Diesel</extra>"))
            ui.style_fig(fig, height=320, yprefix="₹")
            ui.chart(fig)

    with right:
        with ui.card("Petrol sensitivity", "Estimated ₹/L across crude price and exchange rate"):
            crude = np.arange(50, 125, 10)
            fx = np.arange(np.floor(ctx.inr_rate) - 8, np.floor(ctx.inr_rate) + 9, 2)
            z = [[estimate_indian_fuel_prices(c, r)["estimated_petrol_inr"] for c in crude] for r in fx]
            fig = go.Figure(go.Heatmap(
                z=z, x=[f"${c}" for c in crude], y=[f"₹{r:.0f}" for r in fx],
                xtype="array",
                colorscale=[[0, "#12324A"], [0.5, "#B07A1B"], [1, "#F04438"]], showscale=False,
                text=[[f"{v:.0f}" for v in row] for row in z], texttemplate="%{text}",
                textfont=dict(size=11, color="#F8FAFC"),
                hovertemplate="Crude %{x}, USD/INR %{y}: ₹%{z:.1f}/L<extra></extra>",
            ))
            ui.style_fig(fig, height=320, legend=False, hovermode="closest", margin=dict(l=4, r=4, t=8, b=4))
            fig.update_xaxes(title=dict(text="Indian basket (USD/bbl)", font=dict(size=11, color=ui.SUBTLE)),
                             showgrid=False, type="category")
            fig.update_yaxes(title=dict(text="USD/INR", font=dict(size=11, color=ui.SUBTLE)), showgrid=False)
            ui.chart(fig)

    with ui.card("Import bill sensitivity",
                 f"Annual change in India's crude import bill (~{INDIA_CRUDE_IMPORTS_MBD} mb/d imports)"):
        moves = [-20, -10, -5, 5, 10, 20]
        impact = [import_bill_impact(m, ctx.inr_rate) for m in moves]
        fig = go.Figure(go.Bar(
            x=[f"{m:+d} $/bbl" for m in moves], y=[i["usd_billion"] for i in impact],
            marker=dict(color=[ui.GREEN if m < 0 else ui.RED for m in moves]),
            text=[f"{i['usd_billion']:+.1f} bn" for i in impact], textposition="outside",
            textfont=dict(color=ui.MUTED), cliponaxis=False,
            customdata=[i["inr_crore"] for i in impact],
            hovertemplate="%{x}: $%{y:.1f} bn (₹%{customdata:,.0f} crore)<extra></extra>",
        ))
        ui.style_fig(fig, height=300, legend=False, hovermode="closest", yprefix="$", ysuffix=" bn")
        peak = max(abs(i["usd_billion"]) for i in impact) * 1.3
        fig.update_yaxes(range=[-peak, peak])
        fig.update_xaxes(type="category")
        ui.chart(fig)
