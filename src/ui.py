"""
Design system: colour tokens, global CSS and reusable HTML components.
Every page builds its interface from these helpers so the look stays
consistent across the platform.
"""

from __future__ import annotations

import html
import re
from contextlib import contextmanager

import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Tokens
# ---------------------------------------------------------------------------
BG = "#0A0E15"
SURFACE = "#111723"
SURFACE_2 = "#172033"
BORDER = "#1F2939"
BORDER_STRONG = "#2B3850"
TEXT = "#E7ECF3"
MUTED = "#9AA4B5"
SUBTLE = "#667085"
GRID = "rgba(148, 163, 184, 0.09)"

ACCENT = "#F5A524"     # crude amber - primary forecast colour
BLUE = "#4F8EF7"       # history
TEAL = "#2DD4BF"       # risk-aware model
VIOLET = "#A78BFA"     # long-term model
SLATE = "#94A3B8"      # naive baseline
GREEN = "#17B26A"
RED = "#F04438"
YELLOW = "#EAAA08"
ORANGE = "#F79009"

PLOTLY_CONFIG = {"displayModeBar": False, "responsive": True}

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {{
  --bg: {BG}; --surface: {SURFACE}; --surface-2: {SURFACE_2};
  --border: {BORDER}; --border-strong: {BORDER_STRONG};
  --text: {TEXT}; --muted: {MUTED}; --subtle: {SUBTLE};
  --accent: {ACCENT}; --blue: {BLUE}; --teal: {TEAL}; --violet: {VIOLET};
  --green: {GREEN}; --red: {RED}; --yellow: {YELLOW};
}}

html, body, .stApp, .stApp p, .stApp li, .stApp label, .stApp input, .stApp textarea,
.stApp button, .stApp table, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5 {{
  font-family: 'Inter', -apple-system, 'Segoe UI', Roboto, sans-serif;
}}
.stApp {{
  background:
    radial-gradient(1100px 520px at 12% -8%, rgba(245,165,36,0.07), transparent 60%),
    radial-gradient(900px 480px at 100% -5%, rgba(79,142,247,0.07), transparent 60%),
    var(--bg);
  color: var(--text);
}}
[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stDecoration"], [data-testid="stAppDeployButton"] {{ display: none; }}
.block-container {{ padding: 2.4rem 2.6rem 4rem; max-width: 1480px; }}
@media (max-width: 640px) {{ .block-container {{ padding: 4rem 1rem 3rem; }} }}

/* Sidebar */
[data-testid="stSidebar"] {{ background: {SURFACE}; border-right: 1px solid var(--border); }}
[data-testid="stSidebarNav"] a, [data-testid="stSidebarNavLink"] {{ border-radius: 8px; }}
[data-testid="stNavSectionHeader"] {{
  text-transform: uppercase; letter-spacing: .09em; font-size: .68rem !important;
  color: var(--subtle) !important; font-weight: 600;
}}
[data-testid="stSidebarHeader"] {{ padding-bottom: .25rem; }}
[data-testid="stLogo"] {{ height: 2.1rem; max-width: 100%; }}

/* Cards: st.container(border=True) that contains a .gc-card-marker */
[data-testid="stVerticalBlockBorderWrapper"]:has(> div > [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"] .gc-card-marker) {{
  background: linear-gradient(180deg, rgba(255,255,255,0.018), rgba(255,255,255,0)) , var(--surface);
  border: 1px solid var(--border) !important; border-radius: 14px;
}}
[data-testid="stElementContainer"]:has(.gc-card-top > .gc-card-marker:only-child) {{ display: none; }}

/* Plotly charts sit flush inside cards */
[data-testid="stPlotlyChart"] {{ border-radius: 10px; }}

/* Widgets */
.stApp [data-baseweb="select"] > div, .stApp [data-baseweb="input"] > div,
.stApp textarea {{ background: var(--surface-2); border-color: var(--border-strong); }}
.stButton > button, .stDownloadButton > button {{
  border-radius: 10px; border: 1px solid var(--border-strong); background: var(--surface-2);
  font-weight: 500; transition: all .15s ease;
}}
.stButton > button:hover, .stDownloadButton > button:hover {{ border-color: var(--accent); color: var(--accent); }}
.stButton > button[kind="primary"] {{ background: var(--accent); color: #1a1205; border-color: var(--accent); }}
.stButton > button[kind="primary"]:hover {{ filter: brightness(1.08); color: #1a1205; }}
[data-testid="stExpander"] details {{ border-color: var(--border); border-radius: 12px; background: var(--surface); }}
[data-testid="stTabs"] [data-baseweb="tab-list"] {{ gap: 4px; border-bottom: 1px solid var(--border); }}
[data-testid="stTabs"] [data-baseweb="tab"] {{ padding: 8px 14px; border-radius: 8px 8px 0 0; }}
[data-testid="stTabs"] [data-baseweb="tab-highlight"] {{ background: var(--accent); }}
[data-testid="stChatMessage"] {{ background: var(--surface); border: 1px solid var(--border); border-radius: 14px; }}
[data-testid="stDataFrame"] {{ border: 1px solid var(--border); border-radius: 10px; }}
[data-testid="stAlert"] {{ border-radius: 12px; }}
hr {{ border-color: var(--border) !important; }}

/* Page header */
.gc-header {{ display:flex; justify-content:space-between; align-items:flex-end; gap:16px; flex-wrap:wrap;
  margin: 0 0 1.4rem 0; }}
.gc-eyebrow {{ color: var(--accent); font-size:.72rem; font-weight:600; letter-spacing:.12em; text-transform:uppercase; }}
.gc-title {{ font-size: 1.85rem; font-weight: 700; color: var(--text); letter-spacing: -0.02em; line-height:1.2; margin-top:.25rem; }}
.gc-sub {{ color: var(--muted); font-size: .95rem; margin-top: .35rem; max-width: 760px; line-height:1.5; }}
.gc-meta {{ display:flex; gap:8px; flex-wrap:wrap; }}

/* Pills */
.gc-pill {{ display:inline-flex; align-items:center; gap:7px; padding:5px 11px; border-radius:999px;
  font-size:.76rem; font-weight:500; color: var(--muted); background: var(--surface); border:1px solid var(--border); white-space:nowrap; }}
.gc-pill i {{ width:7px; height:7px; border-radius:50%; background: var(--subtle); display:inline-block; }}
.gc-pill.live i {{ background: var(--green); box-shadow: 0 0 0 3px rgba(23,178,106,.18); }}
.gc-pill.warn i {{ background: var(--yellow); box-shadow: 0 0 0 3px rgba(234,170,8,.18); }}
.gc-pill.accent {{ color: var(--accent); border-color: rgba(245,165,36,.35); background: rgba(245,165,36,.08); }}

/* KPI tiles */
.gc-kpis {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin-bottom: 1rem; }}
.gc-kpi {{ background: linear-gradient(180deg, rgba(255,255,255,0.025), rgba(255,255,255,0)), var(--surface);
  border:1px solid var(--border); border-radius:14px; padding:15px 17px 14px; position:relative; overflow:hidden; }}
.gc-kpi::before {{ content:""; position:absolute; left:0; top:0; bottom:0; width:3px; background: var(--kpi-accent, transparent); }}
.gc-kpi-label {{ color: var(--muted); font-size:.74rem; font-weight:500; letter-spacing:.04em; text-transform:uppercase; }}
.gc-kpi-value {{ color: var(--text); font-size:1.55rem; font-weight:650; letter-spacing:-0.02em; margin-top:6px;
  font-variant-numeric: tabular-nums; line-height:1.15; }}
.gc-kpi-unit {{ color: var(--subtle); font-size:.85rem; font-weight:500; margin-left:3px; }}
.gc-kpi-foot {{ display:flex; align-items:center; gap:8px; margin-top:8px; flex-wrap:wrap; }}
.gc-kpi-sub {{ color: var(--subtle); font-size:.76rem; }}
.gc-delta {{ font-size:.76rem; font-weight:600; padding:2px 7px; border-radius:6px; font-variant-numeric: tabular-nums; }}
.gc-delta.up {{ color: var(--green); background: rgba(23,178,106,.12); }}
.gc-delta.down {{ color: var(--red); background: rgba(240,68,56,.12); }}
.gc-delta.neutral {{ color: var(--muted); background: rgba(148,163,184,.12); }}

/* Card header */
.gc-card-head {{ display:flex; justify-content:space-between; align-items:flex-start; gap:12px; margin-bottom:.15rem; }}
.gc-card-title {{ font-size: 1rem; font-weight: 600; color: var(--text); }}
.gc-card-sub {{ font-size:.82rem; color: var(--subtle); margin-top:2px; line-height:1.45; }}

/* Stat list */
.gc-stats {{ display:flex; flex-direction:column; gap:0; }}
.gc-stat {{ display:flex; justify-content:space-between; align-items:center; padding:10px 0; border-bottom:1px solid var(--border); font-size:.88rem; }}
.gc-stat:last-child {{ border-bottom:none; }}
.gc-stat span:first-child {{ color: var(--muted); }}
.gc-stat span:last-child {{ color: var(--text); font-weight:600; font-variant-numeric: tabular-nums; }}

/* News cards */
.gc-news {{ display:flex; justify-content:space-between; gap:18px; padding:14px 16px; border-radius:12px;
  background: var(--surface); border:1px solid var(--border); margin-bottom:10px; text-decoration:none !important;
  transition: border-color .15s ease, transform .15s ease; }}
.gc-news:hover {{ border-color: var(--border-strong); transform: translateY(-1px); }}
.gc-news-title {{ color: var(--text) !important; font-weight:550; font-size:.93rem; line-height:1.45; }}
.gc-news-meta {{ color: var(--subtle); font-size:.77rem; margin-top:6px; display:flex; gap:6px; flex-wrap:wrap; align-items:center; }}
.gc-chip {{ font-size:.7rem; padding:2px 8px; border-radius:6px; background: var(--surface-2); color: var(--muted); border:1px solid var(--border); }}
.gc-news-side {{ display:flex; flex-direction:column; align-items:flex-end; gap:8px; min-width:118px; }}
.gc-dir {{ font-size:.74rem; font-weight:600; padding:3px 9px; border-radius:999px; white-space:nowrap; }}
.gc-dir.bullish {{ color: var(--green); background: rgba(23,178,106,.12); }}
.gc-dir.bearish {{ color: var(--red); background: rgba(240,68,56,.12); }}
.gc-dir.neutral {{ color: var(--muted); background: rgba(148,163,184,.12); }}
.gc-riskbar {{ display:flex; align-items:center; gap:8px; color: var(--muted); font-size:.75rem; font-variant-numeric: tabular-nums; }}
.gc-riskbar b {{ display:block; width:64px; height:5px; border-radius:99px; background: var(--surface-2); overflow:hidden; }}
.gc-riskbar b i {{ display:block; height:100%; border-radius:99px; }}

/* Pipeline / feature tiles */
.gc-steps {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 12px; }}
.gc-step {{ background: var(--surface); border:1px solid var(--border); border-radius:14px; padding:16px; }}
.gc-step-n {{ font-size:.7rem; font-weight:700; letter-spacing:.1em; color: var(--step-color, var(--accent)); text-transform:uppercase; }}
.gc-step-t {{ font-size:.95rem; font-weight:600; color: var(--text); margin-top:6px; }}
.gc-step-d {{ font-size:.82rem; color: var(--muted); margin-top:6px; line-height:1.5; }}

/* Callout */
.gc-callout {{ border:1px solid var(--border); border-left:3px solid var(--accent); background: var(--surface);
  border-radius: 10px; padding: 12px 14px; color: var(--muted); font-size:.86rem; line-height:1.55; }}
.gc-callout b {{ color: var(--text); }}

/* Big number */
.gc-big {{ font-size: 2.6rem; font-weight: 700; letter-spacing:-0.03em; line-height:1; font-variant-numeric: tabular-nums; }}
.gc-muted {{ color: var(--muted); }}
.gc-subtle {{ color: var(--subtle); font-size:.8rem; }}

/* Boot screen */
.gc-boot {{ max-width: 560px; margin: 12vh auto 0; text-align:center; }}
.gc-boot h2 {{ font-size:1.6rem; font-weight:700; color: var(--text); margin: 16px 0 6px; }}
.gc-boot p {{ color: var(--muted); }}

@media (max-width: 640px) {{
  .gc-title {{ font-size: 1.45rem; }}
  .gc-news {{ flex-direction: column; }}
  .gc-news-side {{ flex-direction:row; align-items:center; justify-content:space-between; min-width:0; }}
}}
</style>
"""


# ---------------------------------------------------------------------------
# Rendering primitives
# ---------------------------------------------------------------------------
def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def html_block(markup: str):
    """
    Render raw HTML. The markup is wrapped in a block-level <div> and its
    newlines collapsed, so Markdown treats it as one raw HTML block and never
    wraps fragments (e.g. a leading <a> or <span>) in paragraphs.
    """
    body = re.sub(r"\n\s*", " ", markup).strip()
    st.markdown(f'<div class="gc-html">{body}</div>', unsafe_allow_html=True)


def esc(text) -> str:
    return html.escape(str(text), quote=True)


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------
def usd(x, decimals=2):
    return f"${x:,.{decimals}f}"


def inr(x, decimals=2):
    return f"₹{x:,.{decimals}f}"


def pct(x, decimals=1, sign=True):
    return f"{x:+.{decimals}f}%" if sign else f"{x:.{decimals}f}%"


def tone_of(x, flat=0.05):
    if x > flat:
        return "up"
    if x < -flat:
        return "down"
    return "neutral"


def delta_html(x, text=None, invert=False):
    tone = tone_of(-x if invert else x)
    arrow = {"up": "▲", "down": "▼", "neutral": "■"}[tone_of(x)]
    return f'<span class="gc-delta {tone}">{arrow} {esc(text if text is not None else pct(x))}</span>'


# ---------------------------------------------------------------------------
# Components
# ---------------------------------------------------------------------------
def pill(text, kind=""):
    return f'<span class="gc-pill {kind}"><i></i>{esc(text)}</span>'


def page_header(eyebrow, title, subtitle="", pills=None):
    meta = "".join(pills or [])
    html_block(
        f"""<div class="gc-header"><div>
        <div class="gc-eyebrow">{esc(eyebrow)}</div>
        <div class="gc-title">{esc(title)}</div>
        {f'<div class="gc-sub">{esc(subtitle)}</div>' if subtitle else ''}
        </div><div class="gc-meta">{meta}</div></div>"""
    )


def tag_html(text, color):
    return (f'<span class="gc-delta" style="color:{color};background:{color}1f">{esc(text)}</span>')


def kpi(label, value, unit="", delta=None, delta_text=None, sub="", accent=None, invert=False, tag=None):
    """Returns the HTML for one KPI tile (render several with ``kpi_grid``).
    ``tag`` is an optional (text, hex colour) status chip shown instead of a delta."""
    foot = ""
    if delta is not None or sub or tag:
        d = delta_html(delta, delta_text, invert) if delta is not None else ""
        if tag:
            d += tag_html(*tag)
        foot = f'<div class="gc-kpi-foot">{d}<span class="gc-kpi-sub">{esc(sub)}</span></div>'
    style = f' style="--kpi-accent:{accent}"' if accent else ""
    unit_html = f'<span class="gc-kpi-unit">{esc(unit)}</span>' if unit else ""
    return (f'<div class="gc-kpi"{style}><div class="gc-kpi-label">{esc(label)}</div>'
            f'<div class="gc-kpi-value">{esc(value)}{unit_html}</div>{foot}</div>')


def kpi_grid(tiles):
    html_block(f'<div class="gc-kpis">{"".join(tiles)}</div>')


@contextmanager
def card(title=None, subtitle=None, right=None):
    """A bordered surface. Everything rendered inside the ``with`` block is in the card."""
    box = st.container(border=True)
    with box:
        head = ""
        if title:
            head = (f'<div class="gc-card-head"><div><div class="gc-card-title">{esc(title)}</div>'
                    + (f'<div class="gc-card-sub">{esc(subtitle)}</div>' if subtitle else "")
                    + f'</div><div>{right or ""}</div></div>')
        # Wrapped in a div so Markdown cannot split the marker into its own <p>.
        html_block(f'<div class="gc-card-top"><span class="gc-card-marker"></span>{head}</div>')
        yield box


def stat_list(rows):
    items = "".join(f'<div class="gc-stat"><span>{esc(k)}</span><span>{esc(v)}</span></div>' for k, v in rows)
    html_block(f'<div class="gc-stats">{items}</div>')


def callout(markup):
    html_block(f'<div class="gc-callout">{markup}</div>')


def steps(items):
    """items: list of (label, title, description, color)."""
    tiles = "".join(
        f'<div class="gc-step" style="--step-color:{c}"><div class="gc-step-n">{esc(n)}</div>'
        f'<div class="gc-step-t">{esc(t)}</div><div class="gc-step-d">{esc(d)}</div></div>'
        for n, t, d, c in items
    )
    html_block(f'<div class="gc-steps">{tiles}</div>')


def news_card(item, show_reasoning=False):
    direction = item.get("direction", "Neutral")
    cls = direction.lower()
    arrow = {"bullish": "▲", "bearish": "▼"}.get(cls, "■")
    risk = float(item.get("risk_score", 0.5))
    color = risk_color(risk)
    chips = "".join(f'<span class="gc-chip">{esc(d)}</span>' for d in item.get("drivers", [])[:3])
    when = time_ago(item.get("published"))
    link = item.get("link") or ""
    tag = "a" if link else "div"
    href = f' href="{esc(link)}" target="_blank" rel="noopener noreferrer"' if link else ""
    reasoning = (f'<div class="gc-news-meta" style="margin-top:8px">{esc(item.get("reasoning", ""))}</div>'
                 if show_reasoning else "")
    return (
        f'<{tag} class="gc-news"{href}><div><div class="gc-news-title">{esc(item["title"])}</div>'
        f'<div class="gc-news-meta"><span>{esc(item.get("source", ""))}</span>'
        + (f'<span>·</span><span>{esc(when)}</span>' if when else "")
        + f'{chips}</div>{reasoning}</div>'
        f'<div class="gc-news-side"><span class="gc-dir {cls}">{arrow} {esc(direction)}</span>'
        f'<span class="gc-riskbar"><b><i style="width:{risk * 100:.0f}%;background:{color}"></i></b>{risk:.2f}</span>'
        f'</div></{tag}>'
    )


def risk_color(score):
    if score >= 0.75:
        return RED
    if score >= 0.6:
        return ORANGE
    if score >= 0.4:
        return YELLOW
    if score >= 0.25:
        return GREEN
    return BLUE


def time_ago(when):
    import datetime as dt

    if when is None:
        return ""
    now = dt.datetime.now(when.tzinfo) if getattr(when, "tzinfo", None) else dt.datetime.now()
    seconds = max(0, (now - when).total_seconds())
    if seconds < 3600:
        return f"{max(1, int(seconds // 60))}m ago"
    if seconds < 86400:
        return f"{int(seconds // 3600)}h ago"
    days = int(seconds // 86400)
    return f"{days}d ago" if days < 14 else when.strftime("%d %b %Y")


# ---------------------------------------------------------------------------
# Plotly
# ---------------------------------------------------------------------------
def style_fig(fig: go.Figure, height=360, legend=True, hovermode="x unified", yprefix="", ysuffix="",
              margin=None):
    fig.update_layout(
        height=height,
        margin=margin or dict(l=4, r=8, t=36 if legend else 12, b=4),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, -apple-system, Segoe UI, sans-serif", color=MUTED, size=12),
        hovermode=hovermode,
        hoverlabel=dict(bgcolor=SURFACE_2, bordercolor=BORDER_STRONG,
                        font=dict(color=TEXT, family="Inter, sans-serif", size=12)),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, xanchor="left",
                    bgcolor="rgba(0,0,0,0)", font=dict(color=MUTED, size=12)),
        xaxis=dict(gridcolor=GRID, zeroline=False, showline=False, tickfont=dict(color=SUBTLE),
                   showspikes=False, automargin=True),
        yaxis=dict(gridcolor=GRID, zeroline=False, showline=False, tickfont=dict(color=SUBTLE),
                   tickprefix=yprefix, ticksuffix=ysuffix, automargin=True),
    )
    return fig


def chart(fig, key=None):
    st.plotly_chart(fig, use_container_width=True, theme=None, config=PLOTLY_CONFIG, key=key)


def band_trace(x, lower, upper, color_rgba, name="80% interval", showlegend=True):
    """Filled band between two series."""
    x = list(x)
    return go.Scatter(
        x=x + x[::-1],
        y=list(upper) + list(lower)[::-1],
        fill="toself",
        fillcolor=color_rgba,
        line=dict(color="rgba(0,0,0,0)"),
        hoverinfo="skip",
        name=name,
        showlegend=showlegend,
    )
