"""
Live news feed
==============
Pulls oil & energy headlines from public Google News RSS searches (no API key
required), de-duplicates them, and scores each one with the
GeopoliticalRiskScorer. Falls back to a curated sample set when offline; the
result's ``live`` flag tells the interface which one it is showing.
"""

from __future__ import annotations

import datetime as dt
from email.utils import parsedate_to_datetime

import requests

try:
    import feedparser
    FEEDPARSER_AVAILABLE = True
except ImportError:  # pragma: no cover
    FEEDPARSER_AVAILABLE = False

RSS_FEEDS = [
    ("Oil prices & OPEC", "https://news.google.com/rss/search?q=oil+prices+OPEC&hl=en-IN&gl=IN&ceid=IN:en"),
    ("Crude oil markets", "https://news.google.com/rss/search?q=crude+oil+market&hl=en-IN&gl=IN&ceid=IN:en"),
    ("India fuel & energy", "https://news.google.com/rss/search?q=india+crude+oil+petrol+diesel&hl=en-IN&gl=IN&ceid=IN:en"),
    ("Energy geopolitics", "https://news.google.com/rss/search?q=oil+supply+sanctions+OR+shipping+OR+conflict&hl=en-IN&gl=IN&ceid=IN:en"),
]

REQUEST_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) EnergyIntelligence/2.0"}

_SAMPLE = [
    ("OPEC+ agrees to extend production cuts amid uncertain demand outlook", "Reuters"),
    ("Crude oil prices climb as Middle East tensions escalate near the Strait of Hormuz", "Bloomberg"),
    ("India's crude import bill rises as Brent holds above $80 per barrel", "Economic Times"),
    ("US shale producers report record output, easing global supply concerns", "CNBC"),
    ("Electric vehicle sales in India cross 2 million, slowing diesel demand growth", "Livemint"),
    ("Russia redirects oil exports to Asia as Western sanctions tighten", "Financial Times"),
    ("Government keeps petrol and diesel prices unchanged for another quarter", "NDTV"),
    ("Recession fears push crude oil lower as inventories build", "CNN Business"),
    ("Saudi Arabia signals willingness to raise output to stabilise markets", "Al Jazeera"),
    ("Houthi drone attacks disrupt tanker traffic in the Red Sea", "BBC News"),
    ("Indian refiners post strong margins on discounted crude purchases", "Moneycontrol"),
    ("Ceasefire talks progress, easing fears of a wider regional conflict", "Reuters"),
]


def _sample_headlines(limit: int) -> list[dict]:
    now = dt.datetime.now(dt.timezone.utc)
    items = []
    for i, (title, source) in enumerate(_SAMPLE[:limit]):
        published = now - dt.timedelta(hours=5 * i + 2)
        items.append({"title": title, "source": source, "link": "", "published": published})
    return items


def _parse_date(entry) -> dt.datetime:
    raw = entry.get("published") or entry.get("updated") or ""
    try:
        parsed = parsedate_to_datetime(raw)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=dt.timezone.utc)
        return parsed
    except (TypeError, ValueError):
        return dt.datetime.now(dt.timezone.utc)


def fetch_live_news(max_headlines: int = 15, timeout: float = 8.0) -> tuple[list[dict], bool]:
    """
    Returns ``(headlines, live)``. Each headline is a dict with
    title, source, link and a timezone-aware ``published`` datetime.
    """
    if not FEEDPARSER_AVAILABLE:
        return _sample_headlines(max_headlines), False

    headlines, seen = [], set()
    for topic, url in RSS_FEEDS:
        try:
            response = requests.get(url, headers=REQUEST_HEADERS, timeout=timeout)
            response.raise_for_status()
            feed = feedparser.parse(response.content)
        except Exception as exc:
            print(f"[news_feed] {topic} feed failed: {exc}")
            continue

        for entry in feed.entries[: max_headlines * 2]:
            title = (entry.get("title") or "").strip()
            source = topic
            # Google News formats titles as "Headline - Publisher".
            if " - " in title:
                title, source = (part.strip() for part in title.rsplit(" - ", 1))
            key = title.lower()[:90]
            if not title or key in seen:
                continue
            seen.add(key)
            headlines.append({
                "title": title,
                "source": source,
                "link": entry.get("link", ""),
                "published": _parse_date(entry),
            })

    if not headlines:
        return _sample_headlines(max_headlines), False

    headlines.sort(key=lambda h: h["published"], reverse=True)
    return headlines[:max_headlines], True


def score_news_feed(nlp_scorer, max_headlines: int = 15) -> dict:
    """Fetches the latest headlines and scores them in a single batch."""
    headlines, live = fetch_live_news(max_headlines=max_headlines)
    batch = nlp_scorer.analyze_multiple([h["title"] for h in headlines])

    scored = []
    for info, nlp in zip(headlines, batch["individual_results"]):
        scored.append({**info, **{k: v for k, v in nlp.items() if k != "headline"}})

    driver_counts: dict[str, int] = {}
    for item in scored:
        for driver in item["drivers"]:
            driver_counts[driver] = driver_counts.get(driver, 0) + 1

    return {
        "headlines": scored,
        "aggregate_risk": batch["aggregate_risk_score"],
        "risk_level": batch["risk_level"],
        "high_risk_count": batch["high_risk_count"],
        "total_analyzed": batch["total_headlines"],
        "bullish": batch["bullish"],
        "bearish": batch["bearish"],
        "neutral": batch["neutral"],
        "drivers": dict(sorted(driver_counts.items(), key=lambda kv: kv[1], reverse=True)),
        "live": live,
        "engine": nlp_scorer.engine_name,
        "timestamp": dt.datetime.now(),
    }


RISK_BANDS = [
    (0.75, "Critical", "#F04438", "Multiple high-impact supply threats in the news flow. Strong upside price risk."),
    (0.60, "Elevated", "#F79009", "Geopolitical tension is building. Prices may carry a risk premium."),
    (0.40, "Moderate", "#EAAA08", "Mixed signals in energy news. No dominant supply shock."),
    (0.25, "Low", "#17B26A", "News flow is calm or easing. Little geopolitical premium expected."),
    (0.00, "Minimal", "#2E90FA", "Very calm news environment. Supply risk premium likely fading."),
]


def get_risk_gauge_data(aggregate_risk: float) -> dict:
    for threshold, level, color, description in RISK_BANDS:
        if aggregate_risk >= threshold:
            return {"level": level, "color": color, "description": description}
    return {"level": "Minimal", "color": "#2E90FA", "description": RISK_BANDS[-1][3]}


if __name__ == "__main__":
    items, is_live = fetch_live_news(max_headlines=8)
    print(f"Fetched {len(items)} headlines (live={is_live})")
    for h in items:
        print(f"  [{h['published']:%Y-%m-%d %H:%M}] {h['title']} - {h['source']}")
