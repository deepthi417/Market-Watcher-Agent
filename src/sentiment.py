"""
Sentiment + Event Classifier
-----------------------------
v1: lightweight keyword-based classifier. No API key or model download
required, so the full pipeline is runnable out of the box.

Upgrade path (noted in README): swap this for a fine-tuned DistilBERT
classifier once you have labeled data - the interface (classify()) stays
the same, so nothing downstream needs to change.
"""

import re

BULLISH_TERMS = [
    "beats", "beat estimates", "raise", "raises", "upgrade", "surge",
    "rally", "record high", "strong", "accumulation", "expansion",
    "approval", "advances", "growth", "up ",
]
BEARISH_TERMS = [
    "misses", "downgrade", "selloff", "sell-off", "plunge", "delay",
    "concern", "anxiety", "lawsuit", "investigation", "risk-off",
    "decline", "drop", "down ",
]
EVENT_KEYWORDS = {
    "earnings": ["earnings", "revenue", "Q1", "Q2", "Q3", "Q4", "quarterly"],
    "regulatory": ["regulat", "SEC", "committee", "bill", "filing", "ETF"],
    "macro": ["Fed", "Treasury", "yields", "inflation", "rate", "risk-off"],
    "rumor": ["reportedly", "sources say", "unclear", "unconfirmed"],
}


def _score_sentiment(text: str) -> str:
    text_lower = text.lower()
    bullish_hits = sum(1 for term in BULLISH_TERMS if term in text_lower)
    bearish_hits = sum(1 for term in BEARISH_TERMS if term in text_lower)

    if bullish_hits > bearish_hits:
        return "bullish"
    if bearish_hits > bullish_hits:
        return "bearish"
    return "neutral"


def _detect_event_type(text: str) -> str:
    for event_type, keywords in EVENT_KEYWORDS.items():
        if any(re.search(kw, text, re.IGNORECASE) for kw in keywords):
            return event_type
    return "unclassified"


def classify(headline: str) -> dict:
    """Returns {sentiment, event_type} for a single headline."""
    return {
        "sentiment": _score_sentiment(headline),
        "event_type": _detect_event_type(headline),
    }


def classify_batch(articles: list[dict]) -> list[dict]:
    """Takes news_tool-style articles, returns them annotated with sentiment/event_type."""
    annotated = []
    for article in articles:
        tags = classify(article["headline"])
        annotated.append({**article, **tags})
    return annotated


if __name__ == "__main__":
    samples = [
        "Apple beats Q3 estimates, iPhone revenue up 12% YoY",
        "Broader tech selloff as Treasury yields climb",
        "Markets flat ahead of Fed commentary",
    ]
    for s in samples:
        print(s, "->", classify(s))
