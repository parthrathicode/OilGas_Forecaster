"""
Market analyst assistant
========================
Answers natural-language questions using the live outputs of every module
(prices, LSTM and XGBoost forecasts, news risk, India estimates, validation).

* **Built-in analyst (default, no API key):** priority-ordered intent
  detection with data-driven answer templates. Anything that looks like a
  news headline is scored by the NLP engine.
* **Gemini (optional):** set the ``GEMINI_API_KEY`` environment variable and
  the assistant forwards questions to Google's Gemini API, grounded with the
  same live context. ``GEMINI_MODEL`` selects the model. On any API error it
  falls back to the built-in analyst.
"""

from __future__ import annotations

import os
import re

import requests

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"


def _pct(new, old):
    return (new / old - 1) * 100 if old else 0.0


class OilMarketChatbot:
    """Intent-based market analyst with optional Gemini back-end."""

    def __init__(self, nlp_scorer=None, context: dict | None = None):
        self.nlp_scorer = nlp_scorer
        self.context = context or {}
        self.history: list[dict] = []

        # Order matters: the first intent with a matching pattern wins, so
        # specific questions ("why are prices rising") come before generic
        # ones ("oil price").
        self.intents = [
            ("greeting", [r"^(hi|hello|hey|greetings|namaste|good (morning|afternoon|evening))\b"], self._greeting),
            ("help", [r"^help\b", r"what can you", r"how (do i|to) use", r"\bcapabilit"], self._help),
            ("why_rising", [r"\bwhy\b.*\b(ris|increas|go(ing)? up|surg|spik|high|bull)",
                            r"\b(reason|driver|cause)s?\b.*\b(ris|high|up)"], self._why_rising),
            ("why_falling", [r"\bwhy\b.*\b(fall|decreas|drop|go(ing)? down|crash|low|bear)",
                             r"\b(reason|driver|cause)s?\b.*\b(fall|low|down|drop)"], self._why_falling),
            ("longterm", [r"long.?term", r"\b(3|three|5|five).?years?\b", r"\b20(27|28|29|30)\b",
                          r"next (few )?years", r"\boutlook\b", r"\bxgboost\b"], self._longterm),
            ("prediction", [r"\bpredict", r"\bforecast", r"next (week|month)", r"\bfuture\b",
                            r"\bwill\b.*\b(rise|fall|go up|go down|increase|decrease|be)\b",
                            r"\b30.?day", r"short.?term", r"\btomorrow\b"], self._prediction),
            ("accuracy", [r"\baccura", r"\breliab", r"\btrust", r"how good", r"\bbacktest",
                          r"\bvalidat", r"\berror", r"\brmse\b", r"\bmape\b"], self._accuracy),
            ("model_info", [r"how\b.*\b(model|ai|system|algorithm|it)\b.*\bwork",
                            r"\b(lstm|finbert|architecture|neural|machine learning)\b",
                            r"\bexplain\b.*\b(model|system)"], self._model_info),
            ("india", [r"\bindia", r"\brupee", r"\binr\b", r"\bpetrol", r"\bdiesel", r"fuel price",
                       r"import bill"], self._india),
            ("opec", [r"\bopec", r"production cut", r"supply cut", r"\bsaudi", r"\bbpd\b"], self._opec),
            ("geopolitical", [r"geopolit", r"\bwar\b", r"\bconflict", r"\bsanction", r"\btension",
                              r"middle.?east", r"\biran\b", r"\brussia", r"\bukraine", r"\bhormuz",
                              r"\brisk\b"], self._geopolitical),
            ("news", [r"\bnews\b", r"\bheadlines?\b", r"what.*(happening|going on)",
                      r"market update"], self._news),
            ("price", [r"\bprice", r"\bcost", r"how much", r"\bcurrent\b", r"\btoday\b",
                       r"\bwti\b", r"\bbrent\b"], self._current_price),
        ]

    # ------------------------------------------------------------------
    @property
    def api_key(self) -> str:
        return os.environ.get("GEMINI_API_KEY", "").strip()

    @property
    def mode(self) -> str:
        return "Gemini" if self.api_key else "Built-in analyst"

    def update_context(self, **kwargs):
        self.context.update(kwargs)

    def respond(self, message: str) -> str:
        message = (message or "").strip()
        if not message:
            return "Ask me anything about crude prices, forecasts, geopolitical risk or India's fuel prices."
        answer = None
        if self.api_key:
            answer = self._call_gemini(message)
        if answer is None:
            answer = self._rule_based(message)
        self.history.append({"role": "user", "message": message})
        self.history.append({"role": "assistant", "message": answer})
        return answer

    def _rule_based(self, message: str) -> str:
        lower = message.lower()
        for _, patterns, handler in self.intents:
            if any(re.search(p, lower) for p in patterns):
                return handler(message)
        if len(message.split()) >= 5 and self.nlp_scorer is not None:
            return self._headline(message)
        return self._unknown(message)

    # ------------------------------------------------------------------
    # Gemini
    # ------------------------------------------------------------------
    def _context_brief(self) -> str:
        c = self.context
        lines = [
            f"WTI crude: ${c.get('wti', 0):.2f}/bbl; Brent: ${c.get('brent', 0):.2f}/bbl.",
            f"USD/INR: {c.get('inr', 0):.2f}. Indian basket (proxy): ${c.get('basket', 0):.2f}/bbl.",
            f"Estimated Delhi petrol: Rs {c.get('petrol', 0):.2f}/L, diesel: Rs {c.get('diesel', 0):.2f}/L.",
        ]
        fc = c.get("forecast")
        if fc is not None and len(fc):
            lines.append(f"LSTM forecast: 7 trading days ${fc[6]:.2f}, 30 trading days ${fc[-1]:.2f}.")
        lt = c.get("longterm")
        if lt:
            lines.append(f"36-month XGBoost outlook: {lt['trend']}, ending near ${lt['end_price']:.2f} "
                         f"({lt['total_change_pct']:+.1f}% vs today).")
        news = c.get("news") or {}
        if news:
            lines.append(f"News risk index: {news.get('aggregate_risk', 0.5):.2f} ({news.get('risk_level', 'n/a')}).")
            for h in news.get("headlines", [])[:6]:
                lines.append(f"- {h['title']} ({h['source']}; {h['direction']}; risk {h['risk_score']:.2f})")
        v = c.get("validation") or {}
        if v:
            lines.append(f"Validation: LSTM 30-day MAPE {v['lstm_mape_30']:.1f}% vs naive "
                         f"{v['naive_mape_30']:.1f}%; direction accuracy {v['lstm_dir_30']:.0f}%.")
        return "\n".join(lines)

    def _call_gemini(self, message: str) -> str | None:
        model = os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        system = (
            "You are the market analyst inside an oil & gas intelligence platform. "
            "Answer concisely and professionally using markdown. Ground answers in this live data "
            "and say when something is an estimate:\n\n" + self._context_brief()
        )
        contents = []
        for turn in self.history[-6:]:
            role = "user" if turn["role"] == "user" else "model"
            contents.append({"role": role, "parts": [{"text": turn["message"]}]})
        contents.append({"role": "user", "parts": [{"text": message}]})
        payload = {"systemInstruction": {"parts": [{"text": system}]}, "contents": contents}
        try:
            resp = requests.post(url, json=payload, timeout=20,
                                 headers={"x-goog-api-key": self.api_key, "Content-Type": "application/json"})
            if resp.status_code != 200:
                return None
            return resp.json()["candidates"][0]["content"]["parts"][0]["text"]
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Intent handlers
    # ------------------------------------------------------------------
    def _greeting(self, _):
        return (
            "Hello! I'm your **oil market analyst**. I can walk you through:\n\n"
            "- live WTI / Brent prices and the 30-day LSTM forecast\n"
            "- the 3-year structural outlook\n"
            "- geopolitical risk in today's news flow\n"
            "- what it all means for petrol and diesel in India\n\n"
            "You can also paste any news headline and I'll score its market impact."
        )

    def _help(self, _):
        return (
            "#### What you can ask\n\n"
            "| Topic | Try asking |\n|---|---|\n"
            "| Prices | *What's the current oil price?* |\n"
            "| Forecast | *Where will crude be in 30 days?* |\n"
            "| Long term | *What's the 3-year outlook?* |\n"
            "| Drivers | *Why are oil prices rising?* |\n"
            "| India | *How does this affect petrol prices in India?* |\n"
            "| OPEC | *How do OPEC+ cuts move prices?* |\n"
            "| Risk | *What's the geopolitical risk right now?* |\n"
            "| News | *Summarise the latest headlines* |\n"
            "| Models | *How does the model work?* / *How accurate is it?* |\n\n"
            "Or paste a headline - e.g. *Tanker attacked near the Strait of Hormuz*."
        )

    def _current_price(self, _):
        c = self.context
        wti, brent, inr = c.get("wti", 0), c.get("brent", 0), c.get("inr", 0)
        return (
            "#### Current prices\n\n"
            "| Benchmark | USD / bbl | INR / bbl |\n|---|---|---|\n"
            f"| WTI crude | {wti:.2f} | {wti * inr:,.0f} |\n"
            f"| Brent crude | {brent:.2f} | {brent * inr:,.0f} |\n"
            f"| Indian basket (proxy) | {c.get('basket', 0):.2f} | {c.get('basket', 0) * inr:,.0f} |\n\n"
            f"Exchange rate: **Rs {inr:.2f} per USD**. Source: Yahoo Finance futures (CL=F, BZ=F)."
        )

    def _prediction(self, _):
        c = self.context
        fc, spot = c.get("forecast"), c.get("wti", 0)
        if fc is None or not len(fc):
            return "The forecast is still loading - try again in a moment."
        d7, d30 = fc[6], fc[-1]
        bias = "higher" if d30 > spot else "lower"
        band = c.get("forecast_band")
        band_txt = f" The 80% prediction interval for day 30 is **${band[0]:.2f} - ${band[1]:.2f}**." if band else ""
        return (
            "#### 30-day WTI forecast (price-only LSTM)\n\n"
            "| Horizon | Forecast | vs today |\n|---|---|---|\n"
            f"| Today | {spot:.2f} | - |\n"
            f"| 7 trading days | {d7:.2f} | {_pct(d7, spot):+.1f}% |\n"
            f"| 30 trading days | {d30:.2f} | {_pct(d30, spot):+.1f}% |\n\n"
            f"The model leans **{bias}** over the next month.{band_txt}\n\n"
            "Open **Short-term Forecast** to switch to the risk-aware model and stress-test it with "
            "different risk levels."
        )

    def _why_rising(self, _):
        news = self.context.get("news") or {}
        return (
            "#### What pushes crude higher\n\n"
            "1. **Supply shocks** - OPEC+ cuts, sanctions, outages or attacks on infrastructure\n"
            "2. **Chokepoint risk** - threats to the Strait of Hormuz, Red Sea or Suez routes\n"
            "3. **Strong demand** - growth in China, India and the US; summer driving season\n"
            "4. **Falling inventories** - US stock draws larger than expected\n"
            "5. **Weaker US dollar** - makes dollar-priced oil cheaper for other buyers\n\n"
            f"Right now the news-risk index reads **{news.get('aggregate_risk', 0.5):.2f} "
            f"({news.get('risk_level', 'n/a')})** with {news.get('bullish', 0)} bullish and "
            f"{news.get('bearish', 0)} bearish headlines in the feed."
        )

    def _why_falling(self, _):
        return (
            "#### What pushes crude lower\n\n"
            "1. **Oversupply** - OPEC+ raising output, record US shale production\n"
            "2. **Demand destruction** - recessions, lockdowns, slowing industrial activity\n"
            "3. **De-escalation** - ceasefires and peace deals remove the risk premium\n"
            "4. **Inventory builds** - stockpiles rising faster than expected\n"
            "5. **Stronger US dollar** and **energy transition** (EVs, renewables)\n\n"
            "Try **Scenario Lab** to see how a demand shock or an output increase would move prices."
        )

    def _india(self, _):
        c = self.context
        pt = c.get("passthrough", {})
        petrol_pt = pt.get("petrol", {}).get("per_usd", 0)
        return (
            "#### India impact\n\n"
            "| Metric | Value |\n|---|---|\n"
            f"| Indian basket (proxy) | ${c.get('basket', 0):.2f}/bbl |\n"
            f"| USD/INR | {c.get('inr', 0):.2f} |\n"
            f"| Cost-based petrol estimate (Delhi) | Rs {c.get('petrol', 0):.2f}/L |\n"
            f"| Cost-based diesel estimate (Delhi) | Rs {c.get('diesel', 0):.2f}/L |\n\n"
            f"Every **$1/bbl** move in crude shifts the petrol cost by about **Rs {petrol_pt:.2f}/L**, and adds "
            "roughly **$1.75 bn a year** to India's crude import bill. India imports ~85% of its crude, so a "
            "weaker rupee amplifies every rise. Retail prices are administered, so oil companies often absorb "
            "short-term swings. See **India Impact** for the full price build-up."
        )

    def _opec(self, _):
        return (
            "#### OPEC+ and prices\n\n"
            "OPEC+ controls roughly 40% of world crude output, so its quota decisions set the supply backdrop.\n\n"
            "| Decision | Typical effect |\n|---|---|\n"
            "| Production cut | Tighter market, prices up (rule of thumb: ~$2-4/bbl per 1 mb/d) |\n"
            "| Output increase | More supply, prices down |\n"
            "| Quota compliance | Stabilising |\n"
            "| Emergency meeting | Volatility spike |\n\n"
            "Use **Scenario Lab** to model a cut or an increase of any size."
        )

    def _geopolitical(self, _):
        news = self.context.get("news") or {}
        drivers = news.get("drivers", {})
        top = ", ".join(f"{k.lower()} ({v})" for k, v in list(drivers.items())[:4]) or "none detected"
        market = self.context.get("market_risk")
        market_txt = f" The options market agrees at **{market:.2f}** (OVX percentile)." if market is not None else ""
        return (
            "#### Geopolitical risk right now\n\n"
            f"News-risk index: **{news.get('aggregate_risk', 0.5):.2f} - {news.get('risk_level', 'n/a')}**."
            f"{market_txt}\n\n"
            f"Most frequent drivers in today's headlines: {top}.\n\n"
            "Key chokepoints to watch: the **Strait of Hormuz** (about a fifth of global oil flows), the "
            "**Red Sea / Bab el-Mandeb**, and sanctions on **Russia** and **Iran**.\n\n"
            "The risk-aware LSTM in **Short-term Forecast** lets you see how its forecast changes with risk."
        )

    def _model_info(self, _):
        return (
            "#### How the platform works\n\n"
            "**1 - News intelligence.** Headlines from Google News RSS are scored by **FinBERT** plus an "
            "energy-domain signal layer -> a 0-1 geopolitical risk index.\n\n"
            "**2 - Short-term forecast.** Two **LSTM** networks (PyTorch) read the last 30 trading days and "
            "forecast the next 30 in one pass. The risk-aware model also reads a market-stress channel "
            "(OVX volatility percentile) that can be driven by the news-risk score.\n\n"
            "**3 - Long-term outlook.** **XGBoost** on monthly data (momentum, volatility, mean-reversion, "
            "seasonality) produces a 36-month path with uncertainty bands.\n\n"
            "**4 - India translation.** Prices are converted into rupees and into cost-based petrol and diesel "
            "estimates.\n\n"
            "Every model is validated out-of-sample - see **Model Validation**."
        )

    def _longterm(self, _):
        lt = self.context.get("longterm")
        if not lt:
            return "The long-term model is still loading."
        yearly = lt.get("yearly")
        rows = ""
        if yearly is not None:
            rows = "".join(f"| {year} | {price:.2f} |\n" for year, price in yearly.items())
        return (
            "#### 36-month outlook (XGBoost)\n\n"
            f"Trend: **{lt['trend']}** - ending near **${lt['end_price']:.2f}** "
            f"({lt['total_change_pct']:+.1f}% vs today).\n\n"
            "| Year | Average forecast (USD/bbl) |\n|---|---|\n" + rows +
            "\nLong-horizon forecasts carry wide uncertainty - see the shaded band in **Long-term Outlook**."
        )

    def _news(self, _):
        news = self.context.get("news") or {}
        items = news.get("headlines", [])[:5]
        if not items:
            return "No headlines loaded yet."
        lines = "".join(
            f"- **{h['title']}** - *{h['source']}* · {h['direction'].lower()}, risk {h['risk_score']:.2f}\n"
            for h in items
        )
        return (
            f"#### Latest headlines\n\nNews-risk index: **{news.get('aggregate_risk', 0.5):.2f} "
            f"({news.get('risk_level', 'n/a')})**\n\n{lines}\nFull scored feed in **News & Risk**."
        )

    def _accuracy(self, _):
        v = self.context.get("validation") or {}
        if not v:
            return "Validation results are still loading."
        verdict = "beats" if v["lstm_skill_30"] > 0 else "does not beat"
        return (
            "#### How accurate are the forecasts?\n\n"
            f"On {v['n_forecasts']:,} out-of-sample forecasts (data the models never trained on):\n\n"
            "| Model | 30-day MAPE | Direction called |\n|---|---|---|\n"
            f"| Price-only LSTM | {v['lstm_mape_30']:.1f}% | {v['lstm_dir_30']:.0f}% |\n"
            f"| Risk-aware LSTM | {v['hybrid_mape_30']:.1f}% | {v['hybrid_dir_30']:.0f}% |\n"
            f"| Naive (no change) | {v['naive_mape_30']:.1f}% | - |\n\n"
            f"The price-only LSTM {verdict} the naive benchmark at 30 days "
            f"(skill {v['lstm_skill_30']:+.1f}%). Oil is notoriously close to a random walk, so the naive "
            "benchmark is the honest bar to clear. Details and crisis tests are in **Model Validation**."
        )

    def _headline(self, message):
        r = self.nlp_scorer.analyze_headline(message)
        drivers = ", ".join(r["drivers"]) or "none"
        return (
            "#### Headline analysis\n\n"
            f"> {message}\n\n"
            "| Signal | Reading |\n|---|---|\n"
            f"| Direction for crude | **{r['direction']}** |\n"
            f"| Impact | {r['impact']} |\n"
            f"| Risk score | {r['risk_score']:.2f} ({r['risk_level']}) |\n"
            f"| FinBERT tone | {r['sentiment']} ({r['confidence']:.0f}%) |\n"
            f"| Drivers | {drivers} |\n\n"
            f"{r['reasoning']}"
        )

    def _unknown(self, _):
        return (
            "I'm not sure I follow. Try asking about **current prices**, the **30-day forecast**, "
            "**India's fuel prices** or **geopolitical risk** - or paste a news headline to score it. "
            "Type `help` for the full list."
        )
