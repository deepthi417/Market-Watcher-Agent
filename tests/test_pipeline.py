import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.price_tool import get_price_history, compute_moves
from src.news_tool import get_news, news_near_timestamp
from src.sentiment import classify, classify_batch
from src.backtest import naive_baseline_call, verify_outcome, VerdictRecord, score_records, compute_metrics


def test_mock_price_history_loads():
    history = get_price_history("AAPL", mode="mock")
    assert len(history) > 0
    assert "close" in history[0] and "timestamp" in history[0]


def test_compute_moves_detects_significant_changes():
    history = get_price_history("AAPL", mode="mock")
    moves = compute_moves(history, min_pct_change=1.0)
    assert all(abs(m["pct_change"]) >= 1.0 for m in moves)


def test_sentiment_classifies_bullish_headline():
    result = classify("Apple beats Q3 estimates, iPhone revenue up 12% YoY")
    assert result["sentiment"] == "bullish"
    assert result["event_type"] == "earnings"


def test_sentiment_classifies_bearish_headline():
    result = classify("Broader tech selloff as Treasury yields climb")
    assert result["sentiment"] == "bearish"


def test_news_near_timestamp_filters_correctly():
    news = get_news("AAPL", mode="mock")
    nearby = news_near_timestamp(news, "2026-09-20T16:00:00Z", window_hours=24)
    assert all(a["timestamp"] <= "2026-09-20T16:00:00Z" for a in nearby)


def test_naive_baseline_call_bullish_majority():
    evidence = [{"sentiment": "bullish"}, {"sentiment": "bullish"}, {"sentiment": "bearish"}]
    assert naive_baseline_call(evidence) == "up"


def test_verify_outcome_continued():
    history = [
        {"timestamp": "t0", "close": 100},
        {"timestamp": "t1", "close": 105},
        {"timestamp": "t2", "close": 108},
        {"timestamp": "t3", "close": 112},
    ]
    move = {"timestamp": "t1", "pct_change": 5.0, "to_price": 105}
    assert verify_outcome(history, move, lookahead_periods=2) == "continued"


def test_verify_outcome_unknown_when_no_future_data():
    history = [{"timestamp": "t0", "close": 100}, {"timestamp": "t1", "close": 105}]
    move = {"timestamp": "t1", "pct_change": 5.0, "to_price": 105}
    assert verify_outcome(history, move, lookahead_periods=2) == "unknown"


def test_compute_metrics_handles_empty_scored_set():
    records = [VerdictRecord(ticker="AAPL", move={"timestamp": "t99", "pct_change": 1.0, "to_price": 100},
                              verdict={"caused_by": "news", "confidence": 0.9}, evidence=[])]
    history = {"AAPL": [{"timestamp": "t0", "close": 100}]}
    scored = score_records(records, history)
    metrics = compute_metrics(scored)
    assert "error" in metrics
