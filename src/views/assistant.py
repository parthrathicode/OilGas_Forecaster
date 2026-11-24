"""AI analyst chat page."""

from __future__ import annotations

import streamlit as st

import ui
from chatbot import OilMarketChatbot
from core import chat_context, get_context

SUGGESTIONS = [
    ("Current prices", "What's the current oil price?"),
    ("30-day forecast", "Where will crude be in 30 days?"),
    ("3-year outlook", "What's the 3-year outlook?"),
    ("Geopolitical risk", "What's the geopolitical risk right now?"),
    ("India impact", "How does this affect petrol prices in India?"),
    ("Latest news", "Summarise the latest headlines"),
    ("Model accuracy", "How accurate are the forecasts?"),
    ("How it works", "How does the model work?"),
]


def _queue(question):
    st.session_state.pending_question = question


def _clear():
    st.session_state.pop("chat", None)
    st.session_state.pop("chatbot", None)


def _md(text):
    # Streamlit renders $...$ as LaTeX; escape dollars so prices display literally.
    return text.replace("$", "\\$")


def render():
    ctx = get_context()
    if "chatbot" not in st.session_state:
        st.session_state.chatbot = OilMarketChatbot(nlp_scorer=ctx.nlp)
    bot: OilMarketChatbot = st.session_state.chatbot
    bot.update_context(**chat_context(ctx))

    if "chat" not in st.session_state:
        st.session_state.chat = [{"role": "assistant", "content": bot.respond("hello")}]

    mode_pill = (ui.pill("Gemini connected", "live") if bot.mode == "Gemini"
                 else ui.pill("Built-in analyst · offline", "accent"))
    ui.page_header(
        "Intelligence",
        "AI analyst",
        "Ask about prices, forecasts, geopolitical risk or India's fuel prices - answers are grounded in the "
        "live numbers on this platform. Paste any headline to have it scored.",
        pills=[mode_pill],
    )

    question = st.chat_input("Ask about prices, forecasts, risk or India…")
    question = question or st.session_state.pop("pending_question", None)
    if question:
        st.session_state.chat.append({"role": "user", "content": question})
        with st.spinner("Analysing…"):
            st.session_state.chat.append({"role": "assistant", "content": bot.respond(question)})

    chat_col, side_col = st.columns([2.4, 1], gap="medium")
    with side_col:
        with ui.card("Suggested questions"):
            for label, prompt in SUGGESTIONS:
                st.button(label, key=f"sugg_{label}", use_container_width=True, on_click=_queue, args=(prompt,))
        with ui.card("Live context"):
            ui.stat_list([
                ("WTI", ui.usd(ctx.spot)),
                ("30-day forecast", ui.usd(ctx.std_path[-1])),
                ("News risk", f"{ctx.news['aggregate_risk']:.2f} · {ctx.news_gauge['level']}"),
                ("Petrol estimate", f"{ui.inr(ctx.india['petrol_estimate_inr'])}/L"),
            ])
        st.button("Clear conversation", icon=":material/delete_sweep:", use_container_width=True, on_click=_clear)
        if bot.mode != "Gemini":
            st.caption("Tip: set the `GEMINI_API_KEY` environment variable before launching to switch the "
                       "analyst to Google Gemini.")

    with chat_col:
        for msg in st.session_state.chat:
            avatar = ":material/person:" if msg["role"] == "user" else ":material/insights:"
            with st.chat_message(msg["role"], avatar=avatar):
                st.markdown(_md(msg["content"]))
