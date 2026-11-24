"""Model validation page - walk-forward hold-out and crisis stress tests."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

import ui
from backtester import MODEL_LABELS
from core import get_context, get_crisis_tests
from views.common import rgba, today_marker

MODEL_COLORS = {"lstm": ui.ACCENT, "hybrid": ui.TEAL, "naive": ui.SLATE}
HORIZON_CHOICES = {"1 day": 1, "5 days": 5, "10 days": 10, "30 days": 30}


def render():
    ctx = get_context()
    bt = ctx.backtest
    wf, info, v = bt.walkforward, bt.test_info, ctx.validation

    ui.page_header(
        "Intelligence",
        "Model validation",
        "Every number on this page comes from data the models never saw during training. Forecasts are "
        "benchmarked against a naive no-change forecast, the standard bar for price models.",
    )

    best = "lstm" if v["lstm_mape_30"] <= v["hybrid_mape_30"] else "hybrid"
    best_skill = v[f"{best}_skill_30"]
    years = (info["test_end"] - info["test_start"]).days / 365.25
    ui.kpi_grid([
        ui.kpi("Unseen test period", f"{years:.1f}", " years",
               sub=f"{info['test_start']:%b %Y} – {info['test_end']:%b %Y}", accent=ui.BLUE),
        ui.kpi("Forecasts evaluated", f"{info['n_forecasts']:,}", "", sub="30-day paths, one per trading day"),
        ui.kpi("Best 30-day MAPE", f"{v[f'{best}_mape_30']:.1f}%", "", best_skill, delta_text=f"{best_skill:+.1f}% skill",
               sub=f"{MODEL_LABELS[best]} vs naive {v['naive_mape_30']:.1f}%", accent=ui.ACCENT),
        ui.kpi("30-day direction called", f"{v['lstm_dir_30']:.0f}%", "", sub="price-only LSTM"),
        ui.kpi("Learned-range coverage", f"{v['hybrid_coverage']:.0f}%", "",
               sub="target 80% · risk-aware LSTM", accent=ui.TEAL),
    ])

    left, right = st.columns([1.5, 1], gap="medium")
    with left:
        with ui.card("Error by forecast horizon", "Mean absolute % error on the hold-out set - lower is better"):
            fig = go.Figure()
            h = np.arange(1, len(wf["naive"]["mape_by_h"]) + 1)
            for key in ("naive", "lstm", "hybrid"):
                fig.add_trace(go.Scatter(
                    x=h, y=wf[key]["mape_by_h"], name=MODEL_LABELS[key], mode="lines",
                    line=dict(color=MODEL_COLORS[key], width=2.4, dash="dash" if key == "naive" else "solid"),
                    hovertemplate="day %{x}: %{y:.2f}%<extra>" + MODEL_LABELS[key] + "</extra>",
                ))
            ui.style_fig(fig, height=330, ysuffix="%")
            fig.update_xaxes(title=dict(text="Trading days ahead", font=dict(size=11, color=ui.SUBTLE)))
            ui.chart(fig)
    with right:
        with ui.card("Interval calibration", "Share of outcomes inside the risk-aware model's learned 80% range"):
            cov = wf["hybrid"]["coverage_by_h"]
            fig = go.Figure()
            fig.add_hrect(y0=75, y1=85, fillcolor=rgba(ui.GREEN, 0.08), line_width=0)
            fig.add_hline(y=80, line=dict(color=ui.GREEN, dash="dot", width=1.2))
            fig.add_trace(go.Scatter(x=np.arange(1, len(cov) + 1), y=cov, mode="lines+markers",
                                     line=dict(color=ui.TEAL, width=2.2), marker=dict(size=4),
                                     name="Coverage", hovertemplate="day %{x}: %{y:.0f}%<extra></extra>"))
            ui.style_fig(fig, height=200, legend=False, ysuffix="%", hovermode="closest",
                         margin=dict(l=4, r=8, t=8, b=4))
            fig.update_yaxes(range=[50, 100])
            ui.chart(fig)
            ui.callout(
                "<b>Reading this honestly.</b> Crude oil trades close to a random walk, so the naive "
                "“tomorrow = today” forecast is very hard to beat on point accuracy. The risk-aware model's "
                "strength is a <b>well-calibrated range</b>: its 80% band really contains ~80% of outcomes."
            )

    with ui.card("Walk-forward scorecard", f"{info['n_forecasts']:,} forecasts on unseen data, "
                                           f"trained {info['train_start']:%b %Y} – {info['train_end']:%b %Y}"):
        st.dataframe(
            bt.walkforward_table(), hide_index=True, use_container_width=True,
            column_config={
                "MAPE day 1 (%)": st.column_config.NumberColumn(format="%.2f"),
                "MAPE day 7 (%)": st.column_config.NumberColumn(format="%.2f"),
                "MAPE day 30 (%)": st.column_config.NumberColumn(format="%.2f"),
                "RMSE day 30 ($)": st.column_config.NumberColumn(format="%.2f"),
                "Direction day 30 (%)": st.column_config.NumberColumn(format="%.1f"),
                "Skill vs naive (%)": st.column_config.NumberColumn(format="%+.1f"),
            },
        )

    with ui.card("Forecast vs actual on the hold-out set", "Each point is a forecast made N days earlier"):
        choice = st.segmented_control("Horizon", list(HORIZON_CHOICES), default="30 days", key="val_h",
                                      label_visibility="collapsed") or "30 days"
        k = HORIZON_CHOICES[choice]
        target_dates = info["dates"][np.searchsorted(info["dates"], info["origin_dates"]) + k]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=target_dates, y=info["actual"][:, k - 1], name="Actual WTI",
                                 line=dict(color=ui.BLUE, width=2),
                                 hovertemplate="$%{y:.2f}<extra>Actual</extra>"))
        for key in ("lstm", "hybrid"):
            fig.add_trace(go.Scatter(x=target_dates, y=wf[key]["paths"][:, k - 1], name=MODEL_LABELS[key],
                                     line=dict(color=MODEL_COLORS[key], width=1.6),
                                     hovertemplate="$%{y:.2f}<extra>" + MODEL_LABELS[key] + "</extra>"))
        ui.style_fig(fig, height=360, yprefix="$")
        ui.chart(fig)

    st.write("")
    ui.html_block(
        '<div class="gc-card-title" style="margin:4px 0 2px">Crisis stress tests</div>'
        '<div class="gc-card-sub" style="margin-bottom:12px">For each shock, fresh models are trained only on '
        'data up to the day before the event and asked to forecast the next 30 trading days blind.</div>'
    )
    with st.spinner("Training pre-crisis models (first run only, about a minute)…"):
        crisis = get_crisis_tests(ctx.key, ctx.prices, ctx.dates, ctx.risk)

    available = {k: r for k, r in crisis.items() if r.get("available")}
    if not available:
        st.info("Not enough price history to run the crisis tests.")
        return

    tabs = st.tabs([r["event"]["short"] for r in available.values()])
    for tab, (key, res) in zip(tabs, available.items()):
        with tab:
            render_crisis(res)

    with st.expander("Methodology"):
        st.markdown(
            "- **Hold-out split.** Models train on the first 80% of the daily history; the last 20% is never "
            "seen during training or early stopping.\n"
            "- **Walk-forward evaluation.** From every trading day in the hold-out period the models forecast "
            "the next 30 trading days, which are then compared with what actually happened.\n"
            "- **Benchmark.** The naive forecast assumes the price stays where it is. Skill = "
            "1 − MAPE(model) ÷ MAPE(naive).\n"
            "- **Intervals.** The price-only model's 80% interval is built from the 10th–90th percentile of its "
            "hold-out errors at each horizon. The risk-aware model learns its own 10th/50th/90th percentiles "
            "with a quantile (pinball) loss; *coverage* shows how often reality landed inside that range.\n"
            "- **Crisis tests.** Separate models are trained on data available the day before each event, so "
            "nothing about the shock leaks into training."
        )


def render_crisis(res):
    event = res["event"]
    st.caption(event["description"])
    rows = res["results"]
    actual_move = res["actual_move"]

    tiles = [ui.kpi("Actual 30-day move", ui.pct(actual_move), "", sub=f"from {ui.usd(res['origin_price'])}",
                    accent=ui.BLUE)]
    for key in ("lstm", "hybrid", "naive"):
        m = rows[key]["metrics"]
        if key == "naive":
            tag = None
        elif abs(m["predicted_move"]) < 1.0:  # under 1% is not a directional call
            tag = ("No clear call", ui.SLATE)
        elif np.sign(m["predicted_move"]) == np.sign(actual_move):
            tag = ("Direction ✓", ui.GREEN)
        else:
            tag = ("Direction ✗", ui.RED)
        tiles.append(ui.kpi(MODEL_LABELS[key], f"{m['mape']:.1f}%", " MAPE", tag=tag,
                            sub=f"called {ui.pct(m['predicted_move'])}" if key != "naive" else "flat forecast",
                            accent=MODEL_COLORS[key]))
    ui.kpi_grid(tiles)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=res["context_dates"], y=res["context_prices"], name="Before the event",
                             line=dict(color=ui.SUBTLE, width=1.8),
                             hovertemplate="$%{y:.2f}<extra>History</extra>"))
    x = [res["context_dates"][-1]] + list(res["forecast_dates"])
    anchor = res["origin_price"]
    fig.add_trace(go.Scatter(x=x, y=[anchor] + list(res["actual"]), name="What happened",
                             line=dict(color=ui.BLUE, width=2.6),
                             hovertemplate="$%{y:.2f}<extra>Actual</extra>"))
    for key in ("lstm", "hybrid", "naive"):
        fig.add_trace(go.Scatter(x=x, y=[anchor] + list(rows[key]["path"]), name=MODEL_LABELS[key],
                                 line=dict(color=MODEL_COLORS[key], width=2, dash="dash" if key == "naive" else "solid"),
                                 hovertemplate="$%{y:.2f}<extra>" + MODEL_LABELS[key] + "</extra>"))
    today_marker(fig, res["origin_date"], "Forecast made")
    ui.style_fig(fig, height=360, yprefix="$")
    ui.chart(fig)
