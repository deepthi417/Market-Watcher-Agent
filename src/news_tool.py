"""
News Tool
---------
Fetches recent headlines for a tracked ticker.

Same mock/live pattern as price_tool.py. Live mode uses NewsAPI
(https://newsapi.org) - free tier is enough for a v1 portfolio project.
"""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Maps tickers to search terms, since "AAPL" alone is a weak news query
TICKER_SEARCH_TERMS = {
    "AAPL": "Apple Inc",
    "BTC": "Bitcoin",
    "ETH": "Ethereum",
}


def _load_mock_news(ticker: str) -> list[dict]:
    with open(DATA_DIR / "mock_news.json") as f:
        all_news = json.load(f)
    if ticker not in all_news:
        raise ValueError(f"No mock news data for ticker '{ticker}'. "
                          f"Available: {list(all_news.keys())}")
    return all_news[ticker]


def _fetch_live_news(ticker: str, days_back: int = 5) -> list[dict]:
    import requests

    api_key = os.getenv("NEWSAPI_KEY")
    if not api_key:
        raise RuntimeError("NEWSAPI_KEY not set - required for live news mode.")

    query = TICKER_SEARCH_TERMS.get(ticker.upper(), ticker)
    from_date = (datetime.utcnow() - timedelta(days=days_back)).strftime("%Y-%m-%d")

    resp = requests.get(
        "https://newsapi.org/v2/everything",
        params={
            "q": query,
            "from": from_date,
            "sortBy": "publishedAt",
            "language": "en",
            "apiKey": api_key,
        },
        timeout=15,
    )
    resp.raise_for_status()
    articles = resp.json().get("articles", [])
    return [
        {
            "timestamp": a["publishedAt"],
            "headline": a["title"],
            "source": a["source"]["name"],
        }
        for a in articles
    ]


def get_news(ticker: str, mode: Literal["mock", "live"] | None = None) -> list[dict]:
    """Returns a list of {timestamp, headline, source} dicts."""
    mode = mode or os.getenv("MARKET_WATCHER_MODE", "mock")

    if mode == "mock":
        return _load_mock_news(ticker)
    return _fetch_live_news(ticker)


def news_near_timestamp(news: list[dict], timestamp: str, window_hours: int = 24) -> list[dict]:
    """
    Filters news to articles published within `window_hours` BEFORE the given
    timestamp (a price move's end time) - this is the candidate evidence set
    handed to the reasoning agent for that specific move.
    """
    target = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    window_start = target - timedelta(hours=window_hours)

    nearby = []
    for article in news:
        published = datetime.fromisoformat(article["timestamp"].replace("Z", "+00:00"))
        if window_start <= published <= target:
            nearby.append(article)
    return nearby


if __name__ == "__main__":
    articles = get_news("AAPL")
    print(f"Loaded {len(articles)} articles for AAPL")
    for a in articles:
        print(a["timestamp"], "-", a["headline"])
