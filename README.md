# GeoCognitive · Energy Intelligence Platform

An AI-powered decision-support platform for crude oil markets. It combines live market data, deep
learning, gradient boosting and financial NLP to forecast WTI crude prices, measure geopolitical risk
in the news flow, and translate everything into what it means for India's fuel prices.

---

## Features

| Area | What it does |
|---|---|
| **Market overview** | Live WTI, Brent, USD/INR and oil-volatility (OVX) data, the 30-day forecast with an 80% interval, top headlines and an India snapshot. |
| **News & risk** | Google News RSS headlines scored by **FinBERT** plus an energy-domain signal layer (conflict, sanctions, chokepoints, OPEC+ cuts, outages vs. de-escalation, extra supply, weak demand). Produces a 0–1 risk index, price direction per headline, and a live headline analyzer. |
| **Short-term forecast** | Two **PyTorch LSTMs** forecast the next 30 trading days in one pass. The *risk-aware* model is a quantile LSTM trained on an OVX market-stress channel; drive it with news risk, market risk or your own value and watch the learned range widen or narrow. |
| **Long-term outlook** | Direct multi-horizon **XGBoost** models on monthly features (momentum, volatility, mean-reversion, drawdown, seasonality) since 2000, shown as a 36-month fan chart with skill-weighted shrinkage toward a naive forecast. |
| **Scenario lab** | What-if analysis: supply shocks (mb/d), demand shocks, geopolitical risk and exchange rate, with presets such as *OPEC+ cut* or *Hormuz disruption*, and the knock-on effect on Indian fuel prices and the import bill. |
| **India impact** | Indian crude basket in rupees, a cost-based build-up of Delhi petrol and diesel prices, pass-through sensitivities, a crude × rupee heatmap, and import-bill sensitivity. |
| **AI analyst** | A chat assistant grounded in the platform's live numbers. Works offline out of the box; set `GEMINI_API_KEY` to use Google Gemini. Paste any headline to have it scored. |
| **Model validation** | Leak-free walk-forward evaluation on the last 20% of history against a naive benchmark, error-by-horizon curves, interval coverage, and crisis stress tests (COVID-19, Russia–Ukraine, OPEC+ 2023) using models trained only on pre-event data. |

---

## Tech stack

- **Language:** Python 3.11
- **Deep learning / ML:** PyTorch (LSTM, quantile regression), XGBoost, scikit-learn
- **NLP:** Hugging Face Transformers – `ProsusAI/finbert`
- **Data:** yfinance (CL=F, BZ=F, ^OVX, INR=X), feedparser + Google News RSS
- **Interface:** Streamlit multipage app, Plotly charts, custom design system

---

## Project structure

```text
OilGas_Forecaster/
├── .streamlit/config.toml     # Dark theme
├── run.bat                    # One-click launcher for Windows
├── requirements.txt
└── src/
    ├── dashboard.py           # Entry point: boot screen, navigation, sidebar status
    ├── core.py                # Cached service layer (memory + on-disk model cache)
    ├── ui.py                  # Design system: tokens, CSS, cards, KPI tiles, chart styling
    ├── nav.py                 # Page registry for cross-page links
    ├── views/                 # One module per page
    │   ├── overview.py  news.py  forecast.py  outlook.py
    │   └── scenarios.py india.py validation.py assistant.py
    ├── market_data.py         # Yahoo Finance downloads with offline fallbacks
    ├── price_predictor.py     # Price-only and risk-aware (quantile) LSTM engines
    ├── longterm_predictor.py  # Direct multi-horizon XGBoost outlook
    ├── backtester.py          # Walk-forward validation and crisis stress tests
    ├── nlp_scorer.py          # FinBERT + energy-domain risk scorer
    ├── news_feed.py           # RSS ingestion and batch scoring
    ├── india_data.py          # Rupee conversion and retail fuel price model
    └── chatbot.py             # AI analyst (built-in or Gemini)
```

---

## Getting started

### Option A – one click (Windows)

Double-click **`run.bat`**. On first run it creates a virtual environment and installs the
dependencies (a few minutes), then opens the dashboard in your browser.

### Option B – manual

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
streamlit run src/dashboard.py
```

Run the command from the project root so the theme in `.streamlit/config.toml` is applied. The app
opens at <http://localhost:8501>.

**First launch:** the app downloads market data and the FinBERT model (~420 MB, cached afterwards)
and trains its models, which takes about a minute. Trained models are cached in `.model_cache/` and
reused until a new trading day arrives, so later launches start in seconds.

### Optional: Gemini-powered analyst

```bash
# Windows PowerShell
$env:GEMINI_API_KEY = "your-key"
# macOS / Linux
export GEMINI_API_KEY="your-key"
```

Optionally set `GEMINI_MODEL` (default `gemini-2.5-flash`). Without a key the built-in analyst is used.

---

## Methodology notes

- **No look-ahead.** Every validation number comes from data the models never saw. Walk-forward tests
  train on the first 80% of history; crisis tests train on data up to the day before each event.
- **Honest benchmarks.** Oil prices are close to a random walk, so every model is compared with a naive
  "no change" forecast and reported as *skill vs naive*.
- **Calibrated uncertainty.** The price-only LSTM's interval comes from its out-of-sample errors; the
  risk-aware LSTM learns its own 10th/50th/90th percentiles, and its coverage is measured.
- **Data cleaning.** The 20-Apr-2020 negative WTI settlement (a contract-expiry anomaly) is replaced
  with the prior close.
- **Offline-safe.** If Yahoo Finance or Google News is unreachable, the app falls back to clearly
  labelled sample data instead of crashing.
- **India estimates** are cost-based approximations of the Delhi price build-up. Actual pump prices
  are administered and can stay unchanged while costs move.

---

## Disclaimer

For research and educational purposes only. Not investment advice.
