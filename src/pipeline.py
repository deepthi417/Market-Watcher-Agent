"""
Pipeline Orchestrator
---------------------
Runs the full loop for a given ticker:
  price history -> detect moves -> news evidence per move -> classify ->
  reasoning agent verdict -> log to MLflow

Then, separately, run_backtest() scores all logged verdicts against actual
subsequent price action and prints the eval metrics.
"""

import mlflow
from dotenv import load_dotenv

load_dotenv() # populates os.environ from .env

from src.price_tool import get_price_history, compute_moves
from src.news_tool import get_news, news_near_timestamp
from src.sentiment import classify_batch
from src.reasoning_agent import get_verdict
from src.backtest import VerdictRecord, score_records, compute_metrics

TRACKED_TICKERS = ["AAPL", "BTC"]


def process_ticker(ticker: str) -> list[VerdictRecord]:
    price_history = get_price_history(ticker)
    news = classify_batch(get_news(ticker))
    moves = compute_moves(price_history)

    records = []
    for move in moves:
        evidence = news_near_timestamp(news, move["timestamp"])
        verdict = get_verdict(ticker, move, evidence)

        with mlflow.start_run(run_name=f"{ticker}_{move['timestamp']}"):
            mlflow.log_params({
                "ticker": ticker,
                "move_pct": move["pct_change"],
                "move_timestamp": move["timestamp"],
                "n_evidence_articles": len(evidence),
            })
            mlflow.log_metrics({"confidence": verdict["confidence"]})
            mlflow.log_dict(verdict, "verdict.json")
            mlflow.log_dict(evidence, "evidence.json")
            mlflow.set_tag("caused_by", verdict["caused_by"])

        records.append(VerdictRecord(ticker=ticker, move=move, verdict=verdict, evidence=evidence))

    return records


def run_pipeline(tickers: list[str] | None = None) -> dict[str, list[VerdictRecord]]:
    tickers = tickers or TRACKED_TICKERS
    return {ticker: process_ticker(ticker) for ticker in tickers}


def run_backtest(records_by_ticker: dict[str, list[VerdictRecord]]) -> dict:
    all_records = [r for records in records_by_ticker.values() for r in records]
    price_history_by_ticker = {t: get_price_history(t) for t in records_by_ticker}

    scored = score_records(all_records, price_history_by_ticker)
    return compute_metrics(scored)


if __name__ == "__main__":
    mlflow.set_experiment("market-watcher-agent")

    results = run_pipeline()
    for ticker, records in results.items():
        print(f"\n=== {ticker}: {len(records)} verdicts ===")
        for r in records:
            print(f"  {r.move['timestamp']} ({r.move['pct_change']:+.2f}%) -> "
                  f"{r.verdict['caused_by']} (conf={r.verdict['confidence']}): {r.verdict['explanation']}")

    print("\n=== Backtest metrics ===")
    metrics = run_backtest(results)
    for k, v in metrics.items():
        print(f"  {k}: {v}")

