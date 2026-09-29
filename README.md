# Market-Watcher Agent

An autonomous agent that tracks a small set of stocks/crypto assets, correlates price movements with news sentiment, and produces **causal explanations** — not just correlations — for why an asset moved. Every call the agent makes is logged and later verified against actual price outcomes, so the project ships with a real, honest backtested evaluation rather than a single vanity accuracy number.

## Why this project

Most sentiment-based market tools stop at "positive news → price went up." This agent is asked to go one step further: given a price move and a set of recent headlines, it must decide whether the move was actually **news-driven, macro-driven, or noise**, state its confidence, and — critically — surface contradicting evidence when its own explanation might be wrong. That reasoning step, plus a rigorous backtest against a naive sentiment baseline, is the core of the project.

## Architecture

```
Price Tool ──┐
             ├──> Reasoning Agent ──> Verdict (causal call + confidence) ──> Backtest Verifier ──> MLflow log
News Tool ───┘
```

**Pipeline stages:**

1. **Price ingestion** — pulls OHLCV data for the tracked tickers on a schedule.
2. **News ingestion** — fetches recent headlines/articles per ticker.
3. **Sentiment + event classification** — tags each article bullish/bearish/neutral and by event type (earnings, macro, regulatory, rumor).
4. **Causal reasoning agent** — given a price move (magnitude + timing) and ranked, sentiment-tagged news, outputs a structured verdict:
   ```json
   {
     "caused_by": "news | macro | noise",
     "confidence": 0.0,
     "explanation": "string",
     "contradicting_evidence": "string"
   }
   ```
5. **Backtest verifier** — re-checks each logged verdict N hours/days later against what the price actually did, and scores it.

## Tech stack

- **Data sources:** Alpha Vantage / yfinance (stocks), CoinGecko (crypto), NewsAPI or a finance RSS feed (news)
- **Reasoning:** LLM API (structured JSON output) for the causal verdict step
- **Sentiment/event classification:** fine-tuned DistilBERT (or a zero-shot LLM classifier as a v1 fallback)
- **Experiment/eval tracking:** MLflow — each verdict is logged as a run, with its later-verified outcome attached
- **Backend:** FastAPI
- **Frontend:** Streamlit dashboard
- **Deployment:** Docker (Dockerfile.api, Dockerfile.frontend, docker-compose.yml)

## Evaluation

The eval is the differentiator of this project, not an afterthought.

- **Baseline:** naive rule — "positive sentiment → price goes up" — no reasoning involved.
- **Agent:** each causal verdict is logged with a timestamp and checked later against the actual subsequent price move.

**Metrics reported:**

| Metric | What it shows |
|---|---|
| Calibration | When the agent states confidence ≥ 0.8, is it right ~80% of the time? |
| Noise precision | Can the agent correctly identify moves that were *not* news-driven, rather than always inventing a story? |
| Accuracy vs. baseline | Does the agent outperform naive sentiment-direction guessing on the same move set? |

A result where the agent is *not* more accurate on raw direction but is meaningfully better at flagging low-confidence/noise cases is treated as a real, reportable finding — not a failure to hide.

## Scope (v1)

Limited to 2 tracked assets (one large-cap stock, one major crypto) to keep the eval set small enough to verify by hand and keep the backtest loop honest. Coverage breadth is a deliberate non-goal for v1.

## Status

🚧 Early build — architecture and eval design finalized, implementation not yet started.

## Setup

```bash
git clone https://github.com/deepthi417/market-watcher-agent.git
cd market-watcher-agent
docker compose up
```

Local (non-Docker) setup and API keys required (Alpha Vantage / CoinGecko / NewsAPI / LLM provider) will be documented here as the pipeline is built out.

## License

MIT
