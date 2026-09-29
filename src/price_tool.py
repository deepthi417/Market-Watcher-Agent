"""
Price Tool
----------
Fetches recent price history for a tracked ticker.

Two modes, controlled by MARKET_WATCHER_MODE env var:
  - "mock" (default): reads from data/mock_prices.json, no network/API key needed.
  - "live": pulls real data via yfinance (stocks) or CoinGecko (crypto).

Kept deliberately simple (a plain list of {timestamp, close} dicts) so the
reasoning agent and backtester don't need to care which mode produced the data.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Literal

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

CRYPTO_TICKERS = {"BTC", "ETH", "SOL"}  # extend as needed


def _load_mock_prices(ticker: str) -> list[dict]:
    with open(DATA_DIR / "mock_prices.json") as f:
        all_prices = json.load(f)
    if ticker not in all_prices:
        raise ValueError(f"No mock price data for ticker '{ticker}'. "
                          f"Available: {list(all_prices.keys())}")
    return all_prices[ticker]


def _fetch_live_stock_prices(ticker: str, period: str = "5d", interval: str = "1h") -> list[dict]:
    import yfinance as yf  # imported lazily so mock mode has zero extra deps

    data = yf.Ticker(ticker).history(period=period, interval=interval)
    return [
        {"timestamp": ts.isoformat(), "close": float(row["Close"])}
        for ts, row in data.iterrows()
    ]


def _fetch_live_crypto_prices(ticker: str, days: int = 5) -> list[dict]:
    import requests

    coingecko_ids = {"BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana"}
    coin_id = coingecko_ids.get(ticker.upper())
    if not coin_id:
        raise ValueError(f"No CoinGecko mapping for '{ticker}'. Add it to coingecko_ids.")

    resp = requests.get(
        f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart",
        params={"vs_currency": "usd", "days": days},
        timeout=15,
    )
    resp.raise_for_status()
    prices = resp.json()["prices"]  # list of [ms_timestamp, price]
    return [
        {"timestamp": datetime.utcfromtimestamp(ms / 1000).isoformat() + "Z", "close": price}
        for ms, price in prices
    ]


def get_price_history(ticker: str, mode: Literal["mock", "live"] | None = None) -> list[dict]:
    """Returns a list of {timestamp, close} dicts, oldest first."""
    mode = mode or os.getenv("MARKET_WATCHER_MODE", "mock")

    if mode == "mock":
        return _load_mock_prices(ticker)

    if ticker.upper() in CRYPTO_TICKERS:
        return _fetch_live_crypto_prices(ticker)
    return _fetch_live_stock_prices(ticker)


def compute_moves(price_history: list[dict], min_pct_change: float = 1.0) -> list[dict]:
    """
    Scans consecutive price points and returns the ones with a meaningful move,
    each as {from, to, pct_change, timestamp} where timestamp is the move's end time.
    This is what gets handed to the reasoning agent as "something to explain".
    """
    moves = []
    for prev, curr in zip(price_history, price_history[1:]):
        pct_change = (curr["close"] - prev["close"]) / prev["close"] * 100
        if abs(pct_change) >= min_pct_change:
            moves.append({
                "from_price": prev["close"],
                "to_price": curr["close"],
                "pct_change": round(pct_change, 2),
                "timestamp": curr["timestamp"],
            })
    return moves


if __name__ == "__main__":
    history = get_price_history("AAPL")
    print(f"Loaded {len(history)} price points for AAPL")
    for move in compute_moves(history):
        print(move)
