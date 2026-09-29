"""
FastAPI backend for the Market-Watcher Agent.

Endpoints:
  GET  /health              - liveness check
  GET  /verdicts/{ticker}   - runs the pipeline for one ticker, returns verdicts
  GET  /backtest            - runs the pipeline for all tracked tickers + backtest metrics
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException

from src.pipeline import process_ticker, run_pipeline, run_backtest, TRACKED_TICKERS

app = FastAPI(title="Market-Watcher Agent API")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/verdicts/{ticker}")
def get_verdicts(ticker: str):
    try:
        records = process_ticker(ticker.upper())
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return {
        "ticker": ticker.upper(),
        "n_moves": len(records),
        "verdicts": [
            {
                "move": r.move,
                "verdict": r.verdict,
                "n_evidence": len(r.evidence),
            }
            for r in records
        ],
    }


@app.get("/backtest")
def get_backtest():
    results = run_pipeline(TRACKED_TICKERS)
    metrics = run_backtest(results)
    return {
        "tickers": TRACKED_TICKERS,
        "metrics": metrics,
    }
