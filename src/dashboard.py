"""
GeoCognitive - Energy Intelligence Platform
===========================================
Streamlit entry point. Run from the project root with:

    streamlit run src/dashboard.py
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="GeoCognitive · Energy Intelligence",
    page_icon=str(Path(__file__).parent / "assets" / "icon.svg"),
    layout="wide",
    initial_sidebar_state="auto",  # open on desktop, collapsed on phones
)

import core  # noqa: E402
import nav  # noqa: E402
import ui  # noqa: E402
from views import assistant, forecast, india, news, outlook, overview, scenarios, validation  # noqa: E402

ASSETS = Path(__file__).parent / "assets"

ui.inject_css()
st.logo(str(ASSETS / "logo.svg"), size="large", icon_image=str(ASSETS / "icon.svg"))


# ---------------------------------------------------------------------------
# First-load boot screen
# ---------------------------------------------------------------------------
def boot():
    holder = st.empty()
    with holder.container():
        ui.html_block(
            """<div class="gc-boot">
            <div class="gc-eyebrow">Energy intelligence platform</div>
            <h2>Preparing your market workspace</h2>
            <p>Fetching live prices and news and loading the AI models. This takes under a minute
            (models are retrained once per trading day); after that every page loads instantly.</p></div>"""
        )
        _, mid, _ = st.columns([1, 2, 1])
        bar = mid.progress(0.0, text="Starting…")

    def progress(i, label):
        bar.progress(i / len(core.BOOT_STEPS), text=f"{label}…")

    try:
        core.get_context(progress)
    except Exception as exc:  # show a clear, recoverable error instead of a stack dump
        holder.empty()
        st.error("The platform could not start. Check your internet connection and try again.")
        with st.expander("Technical details"):
            st.exception(exc)
        if st.button("Retry", type="primary"):
            st.cache_data.clear()
            st.rerun()
        st.stop()
    holder.empty()
    st.session_state.booted = True


if not st.session_state.get("booted"):
    boot()


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------
def page(name, fn, title, icon, **kwargs):
    return nav.register(name, st.Page(fn, title=title, icon=icon, url_path=name, **kwargs))


pages = {
    "Markets": [
        page("overview", overview.render, "Overview", ":material/space_dashboard:", default=True),
        page("news", news.render, "News & Risk", ":material/public:"),
    ],
    "Forecasting": [
        page("forecast", forecast.render, "Short-term Forecast", ":material/show_chart:"),
        page("outlook", outlook.render, "Long-term Outlook", ":material/timeline:"),
        page("scenarios", scenarios.render, "Scenario Lab", ":material/tune:"),
    ],
    "India": [
        page("india", india.render, "India Impact", ":material/currency_rupee:"),
    ],
    "Intelligence": [
        page("assistant", assistant.render, "AI Analyst", ":material/forum:"),
        page("validation", validation.render, "Model Validation", ":material/verified:"),
    ],
}

current = st.navigation(pages)


# ---------------------------------------------------------------------------
# Sidebar status panel
# ---------------------------------------------------------------------------
def sidebar_status():
    ctx = core.get_context()
    live_count = sum(ctx.live.values())
    market_pill = (ui.pill("Market data live", "live") if ctx.all_live
                   else ui.pill(f"Market data {live_count}/4 live", "warn"))
    news_pill = (ui.pill("News feed live", "live") if ctx.news["live"]
                 else ui.pill("News: offline sample", "warn"))
    nlp_pill = ui.pill("FinBERT loaded", "live") if ctx.nlp.available else ui.pill("FinBERT offline", "warn")
    with st.sidebar:
        st.divider()
        ui.html_block(
            f"""<div style="display:flex;flex-direction:column;gap:8px;align-items:flex-start">
            {market_pill}{news_pill}{nlp_pill}
            <div class="gc-subtle" style="margin-top:4px">Prices as of {ctx.last_date:%d %b %Y} ·
            news scored {ctx.news['timestamp']:%H:%M}</div></div>"""
        )
        c1, c2 = st.columns(2)
        if c1.button("News", icon=":material/refresh:", use_container_width=True,
                     help="Fetch and score the latest headlines"):
            core.refresh_news()
            st.rerun()
        if c2.button("Prices", icon=":material/sync:", use_container_width=True,
                     help="Re-download market data (models retrain only if there is a new trading day)"):
            core.refresh_market_data()
            st.rerun()


sidebar_status()
current.run()
