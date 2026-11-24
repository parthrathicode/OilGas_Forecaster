"""
Geopolitical risk scoring - FinBERT + energy-domain signals
===========================================================

Each headline is scored by two complementary layers:

1. **FinBERT** (ProsusAI/finbert), a BERT model fine-tuned on financial text,
   provides calibrated positive / negative / neutral tone probabilities.
2. **An energy-domain signal layer** recognises the vocabulary that actually
   moves crude: supply-risk drivers (conflict, attacks, sanctions, shipping
   chokepoints, production cuts, outages) versus easing drivers (ceasefires,
   output increases, demand weakness, oversupply), plus reported price moves.

Why both? Generic financial sentiment alone cannot tell "Oil falls on weak
demand" (bearish) from "Tanker attacked in the Strait of Hormuz" (bullish)
- both read as negative news. The domain layer supplies the direction, and
FinBERT's tone sharpens the intensity of the risk score.
"""

from __future__ import annotations

import os
import re
import warnings

import numpy as np

# The global environment may contain a broken TensorFlow install; make sure
# transformers only ever looks at PyTorch.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore", category=FutureWarning)

MODEL_NAME = "ProsusAI/finbert"

# --------------------------------------------------------------------------
# Domain lexicon
# --------------------------------------------------------------------------
SUPPLY_RISK = {
    "Conflict": r"\b(wars?|conflicts?|clash(es)?|fighting|invasion|invade[sd]?|military|troops|hostilit\w+)\b",
    "Attacks": r"\b(attack(s|ed)?|airstrikes?|strikes? on|drones?|missiles?|bomb(ing|ed|s)?|explosions?|sabotage)\b",
    "Sanctions": r"\b(sanction(s|ed)?|embargo(es)?|price cap|export ban|blacklist\w*)\b",
    "Chokepoints": r"\b(hormuz|red sea|suez|bab[- ]el[- ]mandeb|houthis?|strait|shipping lanes?)\b",
    "Tensions": r"\b(tensions?|escalat\w*|threat(s|en\w*)?|crisis|unrest|turmoil|instability|retaliat\w*)\b",
    "Supply cuts": r"\b((output|production|supply) cuts?|cuts? (output|production|supply)|"
                   r"curb(s|ed|ing)? (output|production|supply)|extends? cuts?|deeper cuts?)\b",
    "Outages": r"\b(outages?|disruptions?|disrupt(s|ed|ing)?|shut(down|s|-in)?|halt(s|ed)?|"
               r"force majeure|hurricane|blockade|seiz(e|ed|ure))\b",
}

EASING = {
    "De-escalation": r"\b(ceasefire|truce|peace (deal|talks|treaty|agreement)|de-?escalat\w*|"
                     r"diplomac\w*|diplomatic (progress|breakthrough|talks))\b",
    "More supply": r"\b((boost|raise|increase|hike|lift)(s|d|ed|ing)? (output|production|supply)|"
                   r"(rising|ample|abundant|growing|higher|excess|plentiful|strong|record) "
                   r"(supply|supplies|output|production)|supply (glut|overhang|outlook)|"
                   r"output (rise|hike|increase)s?|unwind\w* cuts?)\b",
    "Weak demand": r"\b(recession|slowdown|weak(er|ening)? demand|demand (concerns?|fears?|worries|slump)|"
                   r"glut|oversuppl\w*|surplus|inventor(y|ies) (build|rise)s?|stockpiles? (rise|build)s?)\b",
}

PRICE_UP = r"\b(rise[sd]?|rising|surge[sd]?|surging|jump(s|ed)?|climb(s|ed)?|rall(y|ies|ied)|" \
           r"gain(s|ed)?|soar(s|ed|ing)?|spike[sd]?|higher|highs?|highest|rebound(s|ed)?)\b"
PRICE_DOWN = r"\b(fall(s|ing)?|fell|drop(s|ped)?|slump(s|ed)?|plunge[sd]?|plunging|tumble[sd]?|" \
             r"slide[sd]?|slip(s|ped)?|decline[sd]?|lower|lows?|lowest|sink(s)?|sank|crash(es|ed)?|" \
             r"retreat(s|ed)?|ease[sd]?|easing|weaker|loss(es)?|pressures?|weighs?)\b"

_SUPPLY_RE = {k: re.compile(v, re.I) for k, v in SUPPLY_RISK.items()}
_EASING_RE = {k: re.compile(v, re.I) for k, v in EASING.items()}
_UP_RE = re.compile(PRICE_UP, re.I)
_DOWN_RE = re.compile(PRICE_DOWN, re.I)

# Simple tone words used only when FinBERT is unavailable (offline, no cache).
_NEG_WORDS = re.compile(r"\b(war|attack|crisis|fears?|risk|threat|loss|slump|plunge|crash|sanction)\w*", re.I)
_POS_WORDS = re.compile(r"\b(gain|growth|record|deal|agree|recover|strong|boost|peace|stabil)\w*", re.I)


def risk_level(score: float) -> str:
    if score >= 0.75:
        return "Critical"
    if score >= 0.60:
        return "Elevated"
    if score >= 0.40:
        return "Moderate"
    if score >= 0.25:
        return "Low"
    return "Minimal"


class GeopoliticalRiskScorer:
    """Scores headlines for oil-market risk and expected price direction."""

    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        self.available = False
        self.error = None
        try:
            from transformers import logging as hf_logging
            from transformers import pipeline

            hf_logging.set_verbosity_error()
            print(f"Loading FinBERT ({model_name})...")
            self.nlp_pipeline = pipeline("text-classification", model=model_name, device=-1)
            self.available = True
            print("FinBERT ready.")
        except Exception as exc:  # offline with no cached weights, etc.
            self.nlp_pipeline = None
            self.error = str(exc)
            print(f"FinBERT unavailable ({exc}); falling back to lexicon-only scoring.")

    @property
    def engine_name(self) -> str:
        return "FinBERT + domain signals" if self.available else "Domain signals (FinBERT offline)"

    # ------------------------------------------------------------------
    def _tone(self, texts: list[str]) -> list[dict]:
        """Positive / negative / neutral probabilities for each text."""
        if self.available:
            outputs = self.nlp_pipeline(texts, top_k=None, truncation=True, batch_size=16)
            if texts and isinstance(outputs, list) and outputs and isinstance(outputs[0], dict):
                outputs = [outputs]  # single input -> list of one
            return [{d["label"].lower(): float(d["score"]) for d in out} for out in outputs]

        tones = []
        for text in texts:
            neg, pos = len(_NEG_WORDS.findall(text)), len(_POS_WORDS.findall(text))
            total = neg + pos + 1
            tones.append({"negative": neg / total, "positive": pos / total, "neutral": 1 / total})
        return tones

    @staticmethod
    def _signals(text: str) -> dict:
        supply = [k for k, rx in _SUPPLY_RE.items() if rx.search(text)]
        easing = [k for k, rx in _EASING_RE.items() if rx.search(text)]
        # Strip matched driver phrases before counting price-move words, so
        # "rising supply" is read as more supply, not as rising prices.
        remainder = text
        for rx in list(_EASING_RE.values()) + list(_SUPPLY_RE.values()):
            remainder = rx.sub(" ", remainder)
        up = len(_UP_RE.findall(remainder))
        down = len(_DOWN_RE.findall(remainder))
        return {"supply": supply, "easing": easing, "up": up, "down": down}

    def _score(self, text: str, tone: dict) -> dict:
        sig = self._signals(text)
        pos, neg, neu = tone.get("positive", 0), tone.get("negative", 0), tone.get("neutral", 0)
        label = max(("positive", pos), ("negative", neg), ("neutral", neu), key=lambda p: p[1])

        # Risk intensity: domain drivers dominate, FinBERT tone adjusts.
        tone_component = 0.5 + 0.5 * (neg - pos)
        domain_component = 0.30 + 0.20 * min(len(sig["supply"]), 3) - 0.12 * min(len(sig["easing"]), 2)
        risk = float(min(0.98, max(0.02, 0.4 * tone_component + 0.6 * domain_component)))

        # Direction: supply risk and reported rises are bullish; easing and falls
        # bearish. A reported price move is the market's own verdict, so it
        # counts double.
        balance = len(sig["supply"]) - len(sig["easing"]) + 2 * (sig["up"] - sig["down"])
        if balance > 0:
            direction = "Bullish"
        elif balance < 0:
            direction = "Bearish"
        else:
            direction = "Neutral"

        if sig["supply"] and direction != "Bearish":
            impact = "Supply risk"
        elif "De-escalation" in sig["easing"]:
            impact = "De-escalation"
        elif {"More supply", "Weak demand"} & set(sig["easing"]):
            impact = "Oversupply / weak demand"
        elif direction == "Bullish":
            impact = "Upward price pressure"
        elif direction == "Bearish":
            impact = "Downward price pressure"
        else:
            impact = "Limited market impact"

        drivers = sig["supply"] + sig["easing"]
        reasoning = self._explain(direction, drivers, label, sig)
        return {
            "headline": text,
            "sentiment": label[0].capitalize(),
            "confidence": round(label[1] * 100, 1),
            "probabilities": {"positive": pos, "negative": neg, "neutral": neu},
            "risk_score": round(risk, 3),
            "risk_level": risk_level(risk),
            "severity": "high" if risk >= 0.6 else ("moderate" if risk >= 0.4 else "low"),
            "direction": direction,
            "impact": impact,
            "drivers": drivers,
            "reasoning": reasoning,
        }

    @staticmethod
    def _explain(direction, drivers, label, sig) -> str:
        parts = []
        if drivers:
            parts.append("Detected " + ", ".join(d.lower() for d in drivers) + " signals.")
        if sig["up"] or sig["down"]:
            moves = []
            if sig["up"]:
                moves.append("rising")
            if sig["down"]:
                moves.append("falling")
            parts.append(f"Headline reports {' and '.join(moves)} prices.")
        parts.append(f"FinBERT reads the tone as {label[0]} ({label[1] * 100:.0f}% confidence).")
        if direction == "Bullish":
            parts.append("Net effect: supply concerns or buying pressure - supportive for crude.")
        elif direction == "Bearish":
            parts.append("Net effect: easing supply risk or softer demand - weighs on crude.")
        else:
            parts.append("Net effect: no clear directional signal for crude.")
        return " ".join(parts)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def analyze_headlines(self, headlines: list[str]) -> list[dict]:
        headlines = [h.strip() for h in headlines if h and h.strip()]
        if not headlines:
            return []
        return [self._score(text, tone) for text, tone in zip(headlines, self._tone(headlines))]

    def analyze_headline(self, headline: str) -> dict:
        return self.analyze_headlines([headline])[0]

    def get_risk_score(self, headline: str) -> float:
        return self.analyze_headline(headline)["risk_score"]

    def analyze_multiple(self, headlines: list[str]) -> dict:
        results = self.analyze_headlines(headlines)
        scores = [r["risk_score"] for r in results]
        aggregate = float(np.mean(scores)) if scores else 0.5
        return {
            "individual_results": results,
            "aggregate_risk_score": round(aggregate, 3),
            "risk_level": risk_level(aggregate),
            "total_headlines": len(results),
            "high_risk_count": sum(1 for r in results if r["severity"] == "high"),
            "bullish": sum(1 for r in results if r["direction"] == "Bullish"),
            "bearish": sum(1 for r in results if r["direction"] == "Bearish"),
            "neutral": sum(1 for r in results if r["direction"] == "Neutral"),
        }


if __name__ == "__main__":
    scorer = GeopoliticalRiskScorer()
    samples = [
        "OPEC+ agrees to extend production cuts into next year",
        "Oil prices surge as Houthi drone attacks disrupt Red Sea shipping",
        "Crude slides as US inventories build and demand fears grow",
        "Ceasefire talks progress, easing supply worries in the Middle East",
        "India announces plans to boost electric vehicle subsidies",
    ]
    for item in scorer.analyze_headlines(samples):
        print(f"{item['risk_score']:.2f}  {item['direction']:8s} {item['impact']:26s} {item['headline']}")
