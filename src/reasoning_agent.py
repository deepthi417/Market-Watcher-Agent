"""
Reasoning Agent
---------------
The core agentic step. Given a price move plus its candidate news evidence,
produces a structured causal verdict:

  {
    "caused_by": "news" | "macro" | "noise",
    "confidence": 0.0-1.0,
    "explanation": "...",
    "contradicting_evidence": "..."
  }

The `contradicting_evidence` field is deliberately required in the prompt -
forcing the model to look for reasons it might be wrong is what turns this
from "summarize the news" into an actual reasoning step, and it's what the
backtest evaluates against later.

Supports two LLM providers - Gemini and Groq (same provider as VulnLens) -
selected automatically based on which API key is set:
  - GEMINI_API_KEY present -> uses Gemini (checked first; Gemini's free tier
    has much higher token/rate limits than Groq's, so it's the default when
    both are available)
  - GROQ_API_KEY present (and no Gemini key) -> uses Groq
  - Neither set -> falls back to a transparent rule-based mock verdict, so
    the pipeline is runnable and demoable with zero setup

Set LLM_PROVIDER=groq or LLM_PROVIDER=gemini to force a specific provider
instead of relying on auto-detection.
"""

import json
import os

SYSTEM_PROMPT = """You are a market-reasoning agent. You are given a price move for an asset \
and a list of news headlines published shortly before that move. Your job is to decide \
whether the move was genuinely caused by the news, driven by broader macro conditions, or \
was likely just noise (no real causal driver).

Be skeptical. Do not assume correlation implies causation. If the news is weak, old, or \
contradicts the direction of the move, say so. You MUST identify contradicting evidence \
if any exists, even if you believe your overall verdict is correct - this is required, \
not optional.

Respond ONLY with a JSON object in this exact shape, no other text:
{
  "caused_by": "news" | "macro" | "noise",
  "confidence": <float 0.0-1.0>,
  "explanation": "<1-2 sentence plain-English explanation>",
  "contradicting_evidence": "<1 sentence on what might contradict this verdict, or 'none found'>"
}"""


def _build_user_prompt(ticker: str, move: dict, evidence: list[dict]) -> str:
    evidence_lines = "\n".join(
        f"- [{a['timestamp']}] ({a.get('sentiment', 'unknown')}, {a.get('event_type', 'unclassified')}) {a['headline']}"
        for a in evidence
    ) or "(no news found in the lookback window)"

    return f"""Ticker: {ticker}
Price move: {move['from_price']} -> {move['to_price']} ({move['pct_change']:+.2f}%) at {move['timestamp']}

Candidate news evidence (published before the move):
{evidence_lines}

Decide what caused this move."""


def _mock_verdict(move: dict, evidence: list[dict]) -> dict:
    """
    Transparent rule-based fallback - NOT a substitute for the real reasoning
    step, just enough to keep the pipeline runnable without an API key.
    Clearly marked so it's never mistaken for a real eval result.
    """
    if not evidence:
        return {
            "caused_by": "noise",
            "confidence": 0.4,
            "explanation": "[MOCK] No news found in the lookback window.",
            "contradicting_evidence": "[MOCK] none found",
        }

    bullish = sum(1 for a in evidence if a.get("sentiment") == "bullish")
    bearish = sum(1 for a in evidence if a.get("sentiment") == "bearish")
    direction_matches = (move["pct_change"] > 0 and bullish > bearish) or \
                         (move["pct_change"] < 0 and bearish > bullish)

    return {
        "caused_by": "news" if direction_matches else "noise",
        "confidence": 0.6 if direction_matches else 0.35,
        "explanation": f"[MOCK] {bullish} bullish / {bearish} bearish articles in window; "
                        f"{'matches' if direction_matches else 'does not clearly match'} move direction.",
        "contradicting_evidence": "[MOCK] rule-based fallback, not real reasoning - set GEMINI_API_KEY or GROQ_API_KEY for real verdicts.",
    }


def _parse_verdict_json(raw: str) -> dict:
    """Shared JSON-parsing + error handling for both providers."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Model didn't return clean JSON - surface this rather than silently failing
        return {
            "caused_by": "noise",
            "confidence": 0.0,
            "explanation": f"[ERROR] Model returned non-JSON output: {raw[:200]}",
            "contradicting_evidence": "n/a",
        }


def _get_verdict_gemini(ticker: str, move: dict, evidence: list[dict], api_key: str) -> dict:
    import google.generativeai as genai

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
        system_instruction=SYSTEM_PROMPT,
    )
    response = model.generate_content(
        _build_user_prompt(ticker, move, evidence),
        generation_config={"temperature": 0.2, "response_mime_type": "application/json"},
    )
    return _parse_verdict_json(response.text)


def _get_verdict_groq(ticker: str, move: dict, evidence: list[dict], api_key: str) -> dict:
    from openai import OpenAI  # Groq exposes an OpenAI-compatible endpoint

    client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
    response = client.chat.completions.create(
        model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(ticker, move, evidence)},
        ],
        temperature=0.2,
    )
    return _parse_verdict_json(response.choices[0].message.content)


def get_verdict(ticker: str, move: dict, evidence: list[dict]) -> dict:
    """
    Main entry point. Returns a verdict dict (see module docstring for shape).

    Provider selection:
      - LLM_PROVIDER env var forces "gemini" or "groq" if set.
      - Otherwise: GEMINI_API_KEY takes priority if present, then GROQ_API_KEY.
      - If neither key is set, falls back to a rule-based mock verdict.
    """
    forced_provider = os.getenv("LLM_PROVIDER", "").lower()
    gemini_key = os.getenv("GEMINI_API_KEY")
    groq_key = os.getenv("GROQ_API_KEY")

    if forced_provider == "gemini" or (not forced_provider and gemini_key):
        if not gemini_key:
            raise RuntimeError("LLM_PROVIDER=gemini set but GEMINI_API_KEY is missing.")
        return _get_verdict_gemini(ticker, move, evidence, gemini_key)

    if forced_provider == "groq" or (not forced_provider and groq_key):
        if not groq_key:
            raise RuntimeError("LLM_PROVIDER=groq set but GROQ_API_KEY is missing.")
        return _get_verdict_groq(ticker, move, evidence, groq_key)

    return _mock_verdict(move, evidence)


if __name__ == "__main__":
    sample_move = {"from_price": 231.50, "to_price": 238.90, "pct_change": 3.2, "timestamp": "2026-09-20T16:00:00Z"}
    sample_evidence = [
        {"timestamp": "2026-09-20T10:15:00Z", "headline": "Apple beats Q3 estimates", "sentiment": "bullish", "event_type": "earnings"},
    ]
    print(json.dumps(get_verdict("AAPL", sample_move, sample_evidence), indent=2))
