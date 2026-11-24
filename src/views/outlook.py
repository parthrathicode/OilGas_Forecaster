"""Long-term (36-month) structural outlook page."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ui
from core import get_context
from longterm_predictor import FEATURE_HELP
from views.common import data_pills, rgba, today_marker

TONE_COLOR = {"up": ui.GREEN, "down": ui.RED, "neutral": ui.YELLOW}


def render():
    ctx = get_context()
    lt, fc, trend = ctx.longterm, ctx.lt_forecast, ctx.lt_trend
    m = lt.training_metrics

    ui.page_header(
        "Forecasting",
        "Long-term outlook",
        "A 36-month structural view from direct multi-horizon gradient-boosted models trained on monthly "
        "market features since 2000. The fan shows the 50% and 80% ranges of outcomes.",
        pills=data_pills(ctx) + [ui.pill(f"{lt.backend} · {m['n_models']} horizon models", "accent")],
    )

    chg = lambda v: (v / ctx.spot - 1) * 100  # noqa: E731
    ui.kpi_grid([
        ui.kpi("Spot (WTI)", ui.usd(ctx.spot), "/bbl", sub=f"{ctx.last_date:%b %Y}", accent=ui.BLUE),
        ui.kpi("12 months", ui.usd(trend["price_12m"]), "", chg(trend["price_12m"]), sub="vs spot"),
        ui.kpi("24 months", ui.usd(trend["price_24m"]), "", chg(trend["price_24m"]), sub="vs spot"),
        ui.kpi("36 months", ui.usd(trend["end_price"]), "", trend["total_change_pct"], sub="vs spot",
               accent=ui.VIOLET),
        ui.kpi("Trend", trend["trend"], "", tag=(f"{trend['total_change_pct']:+.1f}% over 3y",
                                                 TONE_COLOR[trend["tone"]])),
    ])

    with ui.card("WTI · monthly history and 36-month fan chart",
                 "Bands are empirical 10–90% and 25–75% ranges of out-of-sample errors"):
        monthly = lt.monthly.loc["2016":]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=monthly.index, y=monthly.values, name="Monthly average", mode="lines",
                                 line=dict(color=ui.BLUE, width=2), fill="tozeroy",
                                 fillcolor="rgba(79,142,247,0.06)",
                                 hovertemplate="$%{y:.2f}<extra>History</extra>"))
        x = [ctx.last_date] + list(fc.index)
        anchor = ctx.spot
        fig.add_trace(ui.band_trace(x, [anchor] + list(fc["Lower"]), [anchor] + list(fc["Upper"]),
                                    rgba(ui.VIOLET, 0.12), "80% range"))
        fig.add_trace(ui.band_trace(x, [anchor] + list(fc["Lower50"]), [anchor] + list(fc["Upper50"]),
                                    rgba(ui.VIOLET, 0.22), "50% range"))
        fig.add_trace(go.Scatter(x=x, y=[anchor] + list(fc["Price"]), name="Central outlook", mode="lines",
                                 line=dict(color=ui.VIOLET, width=2.6),
                                 hovertemplate="$%{y:.2f}<extra>Outlook</extra>"))
        today_marker(fig, ctx.last_date, "Today")
        ui.style_fig(fig, height=420, yprefix="$")
        ui.chart(fig)

    left, right = st.columns([1.25, 1], gap="medium")
    with left:
        with ui.card("What drives the model", "Relative importance (gain) of each engineered feature"):
            imp = pd.DataFrame(lt.get_feature_importance(), columns=["Feature", "Importance"])
            fig = go.Figure(go.Bar(
                x=imp["Importance"], y=imp["Feature"], orientation="h",
                marker=dict(color=imp["Importance"], colorscale=[[0, "#2B3850"], [1, ui.VIOLET]]),
                hovertemplate="%{y}: %{x:.3f}<extra></extra>",
            ))
            ui.style_fig(fig, height=380, legend=False, hovermode="closest")
            fig.update_yaxes(autorange="reversed", tickfont=dict(color=ui.MUTED))
            fig.update_xaxes(showticklabels=False)
            ui.chart(fig)

    with right:
        with ui.card("Model card", "Time-series cross-validation with a horizon gap"):
            ui.stat_list([
                ("Algorithm", f"{lt.backend}, direct multi-horizon"),
                ("Training data", f"{m['first_month']:%b %Y} – {m['last_month']:%b %Y}"),
                ("Monthly samples · features", f"{m['n_samples']} · {m['n_features']}"),
                ("12-month RMSE (log-return) vs naive", f"{m['cv_rmse_12m']:.3f} vs {m['cv_rmse_naive_12m']:.3f}"),
                ("Average weight on model vs naive", f"{m['avg_weight'] * 100:.0f}%"),
            ])
        with ui.card("Annual averages"):
            yearly = trend["yearly"]
            ui.stat_list([(str(year), ui.usd(price)) for year, price in yearly.items()])

    with ui.card("Where the model adds value",
                 "Out-of-sample skill vs the naive forecast, and the weight the outlook gives the model"):
        per_h = m["per_horizon"]
        hs = list(per_h)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=[f"{h}m" for h in hs], y=[per_h[h]["weight"] * 100 for h in hs],
                             name="Weight on model", marker=dict(color=rgba(ui.VIOLET, 0.55)),
                             hovertemplate="%{x}: %{y:.0f}% model<extra></extra>"))
        fig.add_trace(go.Scatter(x=[f"{h}m" for h in hs], y=[per_h[h]["skill"] for h in hs],
                                 name="Skill vs naive", mode="lines+markers", yaxis="y2",
                                 line=dict(color=ui.ACCENT, width=2.2), marker=dict(size=6),
                                 hovertemplate="%{x}: %{y:+.1f}% skill<extra></extra>"))
        ui.style_fig(fig, height=280, ysuffix="%", hovermode="x unified")
        fig.update_layout(yaxis2=dict(overlaying="y", side="right", ticksuffix="%", showgrid=False,
                                      tickfont=dict(color=ui.SUBTLE), zeroline=False))
        ui.chart(fig)
        ui.callout(
            "Long-horizon oil prices are close to a random walk. Each horizon's prediction is shrunk toward "
            "the naive forecast by the weight that minimised cross-validated error, so the outlook only "
            "departs from today's price where the model has <b>demonstrated</b> skill. The wide fan is the "
            "honest message: plan for a range, not a point."
        )

    with st.expander("Feature guide"):
        for name, text in FEATURE_HELP.items():
            st.markdown(f"**{name}** - {text}")
        st.caption("Long-horizon forecasts are inherently uncertain: treat the path as a structural "
                   "central case, and the band as the range of plausible outcomes.")
