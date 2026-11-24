"""Chart builders and widgets reused across pages."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ui

RANGES = {"3M": 63, "6M": 126, "1Y": 252, "3Y": 756, "5Y": 1260, "Max": None}


def range_picker(key, default="6M", options=("3M", "6M", "1Y", "3Y", "5Y")):
    choice = st.segmented_control("History", list(options), default=default, key=key,
                                  label_visibility="collapsed")
    return RANGES[choice or default]


def today_marker(fig, x, label="Today"):
    fig.add_shape(type="line", x0=x, x1=x, y0=0, y1=1, yref="paper",
                  line=dict(color="rgba(231,236,243,0.28)", width=1, dash="dot"))
    fig.add_annotation(x=x, y=0, yref="paper", text=label, showarrow=False, yanchor="bottom",
                       xanchor="left", xshift=5, yshift=4, font=dict(color=ui.MUTED, size=11))


def forecast_figure(history: pd.Series, dates, path, lower=None, upper=None, color=ui.ACCENT,
                    name="Forecast", band_name="80% interval", height=380):
    """History line + forecast path (+ optional band), joined at today's price."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=history.index, y=history.values, name="WTI history", mode="lines",
        line=dict(color=ui.BLUE, width=2), fill="tozeroy", fillcolor="rgba(79,142,247,0.06)",
        hovertemplate="$%{y:.2f}<extra>WTI</extra>",
    ))
    x = [history.index[-1]] + list(dates)
    anchor = float(history.iloc[-1])
    if lower is not None and upper is not None:
        rgba = _rgba(color, 0.16)
        fig.add_trace(ui.band_trace(x, [anchor] + list(lower), [anchor] + list(upper), rgba, band_name))
    fig.add_trace(go.Scatter(
        x=x, y=[anchor] + list(path), name=name, mode="lines",
        line=dict(color=color, width=2.6), hovertemplate="$%{y:.2f}<extra>" + name + "</extra>",
    ))
    today_marker(fig, history.index[-1])
    ui.style_fig(fig, height=height, yprefix="$")
    lo = min(history.min(), min(lower) if lower is not None else min(path))
    hi = max(history.max(), max(upper) if upper is not None else max(path))
    pad = (hi - lo) * 0.08
    fig.update_yaxes(range=[lo - pad, hi + pad])
    return fig


def _rgba(hex_color, alpha):
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


rgba = _rgba


def data_pills(ctx):
    pills = []
    if ctx.all_live:
        pills.append(ui.pill(f"Live · Yahoo Finance · {ctx.last_date:%d %b %Y}", "live"))
    else:
        pills.append(ui.pill("Some series offline - showing estimates", "warn"))
    return pills
