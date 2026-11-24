"""News intelligence & geopolitical risk page."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

import core
import ui
from core import get_context
from news_feed import RISK_BANDS
from views.common import rgba

FILTERS = ["All", "Bullish", "Bearish", "High risk"]


def gauge_figure(value, color):
    steps = []
    bands = sorted(RISK_BANDS, key=lambda b: b[0])
    for i, (threshold, _, band_color, _) in enumerate(bands):
        upper = bands[i + 1][0] if i + 1 < len(bands) else 1.0
        steps.append({"range": [threshold * 100, upper * 100], "color": rgba(band_color, 0.15)})
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value * 100,
        number={"font": {"color": ui.TEXT, "size": 44, "family": "Inter"}, "valueformat": ".0f"},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": ui.SUBTLE, "tickfont": {"color": ui.SUBTLE, "size": 10}},
            "bar": {"color": color, "thickness": 0.28},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": steps,
        },
    ))
    ui.style_fig(fig, height=230, legend=False, margin=dict(l=24, r=24, t=16, b=0))
    return fig


def render():
    ctx = get_context()
    news, gauge = ctx.news, ctx.news_gauge
    items = news["headlines"]

    source_pill = (ui.pill(f"Live · Google News · scored {news['timestamp']:%H:%M}", "live") if news["live"]
                   else ui.pill("Offline sample headlines", "warn"))
    ui.page_header(
        "Markets",
        "News & geopolitical risk",
        "Energy headlines scored for supply risk and price direction. FinBERT reads the tone; an "
        "energy-domain layer recognises conflict, sanctions, chokepoints, OPEC+ policy and demand signals.",
        pills=[source_pill, ui.pill(news["engine"], "accent")],
    )

    top_driver = next(iter(news["drivers"]), "-")
    ui.kpi_grid([
        ui.kpi("News risk index", f"{news['aggregate_risk']:.2f}", "", tag=(gauge["level"], gauge["color"]),
               sub="0 calm · 1 extreme", accent=gauge["color"]),
        ui.kpi("Headlines scored", str(news["total_analyzed"]), "", sub=f"{news['high_risk_count']} high risk"),
        ui.kpi("Bullish signals", str(news["bullish"]), "", sub="supportive for crude", accent=ui.GREEN),
        ui.kpi("Bearish signals", str(news["bearish"]), "", sub="weighing on crude", accent=ui.RED),
        ui.kpi("Top driver", top_driver, "", sub=f"in {news['drivers'].get(top_driver, 0)} headlines"
               if news["drivers"] else "no strong drivers"),
    ])

    c1, c2, c3 = st.columns([1.05, 1, 1.1], gap="medium")
    with c1:
        with ui.card("Risk gauge", gauge["description"]):
            ui.chart(gauge_figure(news["aggregate_risk"], gauge["color"]))
    with c2:
        with ui.card("Direction mix", "Net price signal per headline"):
            labels = ["Bullish", "Neutral", "Bearish"]
            values = [news["bullish"], news["neutral"], news["bearish"]]
            fig = go.Figure(go.Pie(
                labels=labels, values=values, hole=0.68, sort=False,
                marker=dict(colors=[ui.GREEN, "#475467", ui.RED], line=dict(color=ui.SURFACE, width=3)),
                textinfo="none", hovertemplate="%{label}: %{value}<extra></extra>",
            ))
            bias = news["bullish"] - news["bearish"]
            label = "Net bullish" if bias > 0 else ("Net bearish" if bias < 0 else "Balanced")
            fig.add_annotation(text=f"<b>{label}</b>", showarrow=False, font=dict(color=ui.TEXT, size=14))
            ui.style_fig(fig, height=230, legend=True, hovermode="closest", margin=dict(l=0, r=0, t=30, b=0))
            ui.chart(fig)
    with c3:
        with ui.card("Risk drivers", "How often each signal appears"):
            drivers = news["drivers"]
            if drivers:
                names, counts = list(drivers.keys())[:7], list(drivers.values())[:7]
                fig = go.Figure(go.Bar(x=counts, y=names, orientation="h",
                                       marker=dict(color=ui.ORANGE, line=dict(width=0)),
                                       hovertemplate="%{y}: %{x}<extra></extra>"))
                ui.style_fig(fig, height=230, legend=False, hovermode="closest",
                             margin=dict(l=4, r=8, t=8, b=4))
                fig.update_yaxes(autorange="reversed", tickfont=dict(color=ui.MUTED))
                fig.update_xaxes(dtick=1)
                ui.chart(fig)
            else:
                st.caption("No domain risk drivers detected in the current headlines.")

    with ui.card("Headline analyzer", "Paste any headline to see how the engine reads it"):
        with st.form("analyzer", border=False):
            a, b = st.columns([5, 1], vertical_alignment="bottom")
            text = a.text_input("Headline", placeholder="e.g. Tanker seized near the Strait of Hormuz as tensions rise",
                                label_visibility="collapsed")
            submitted = b.form_submit_button("Analyze", type="primary", use_container_width=True)
        if submitted and text.strip():
            result = ctx.nlp.analyze_headline(text.strip())
            ui.html_block(ui.news_card({**result, "title": result["headline"], "source": "Your input",
                                        "published": None}, show_reasoning=True))

    st.write("")
    head_l, head_r = st.columns([3, 2], vertical_alignment="bottom")
    head_l.markdown('<div class="gc-card-title">Scored feed</div>', unsafe_allow_html=True)
    choice = head_r.segmented_control("Filter", FILTERS, default="All", key="news_filter",
                                      label_visibility="collapsed") or "All"
    if choice == "Bullish":
        shown = [h for h in items if h["direction"] == "Bullish"]
    elif choice == "Bearish":
        shown = [h for h in items if h["direction"] == "Bearish"]
    elif choice == "High risk":
        shown = sorted([h for h in items if h["risk_score"] >= 0.6], key=lambda h: -h["risk_score"])
    else:
        shown = items
    if shown:
        ui.html_block("".join(ui.news_card(h, show_reasoning=False) for h in shown))
    else:
        st.caption("No headlines match this filter.")

    if st.button("Refresh headlines", icon=":material/refresh:"):
        core.refresh_news()
        st.rerun()
