"""
Backtest Verifier
-----------------
This is the module that makes the project defensible rather than a toy demo.

For each logged verdict, checks what the price actually did in the period
AFTER the verdict was made, and scores whether the verdict's implied
direction held up. Also runs a naive sentiment-only baseline on the same
move set so the agent's reasoning can be honestly compared against it.
"""

from dataclasses import dataclass, field


@dataclass
class VerdictRecord:
    ticker: str
    move: dict
    verdict: dict
    evidence: list[dict]
    outcome: str | None = field(default=None)  # "continued" | "reversed" | "unknown"


def naive_baseline_call(evidence: list[dict]) -> str:
    """
    The baseline the agent must beat: pure sentiment counting, no reasoning.
    Returns 'up', 'down', or 'flat'.
    """
    bullish = sum(1 for a in evidence if a.get("sentiment") == "bullish")
    bearish = sum(1 for a in evidence if a.get("sentiment") == "bearish")
    if bullish > bearish:
        return "up"
    if bearish > bullish:
        return "down"
    return "flat"


def verify_outcome(price_history: list[dict], move: dict, lookahead_periods: int = 2) -> str:
    """
    Finds the move's timestamp in price_history, then checks the direction of
    price change over the next `lookahead_periods` points.
    Returns "continued", "reversed", or "unknown" (not enough future data).
    """
    timestamps = [p["timestamp"] for p in price_history]
    if move["timestamp"] not in timestamps:
        return "unknown"

    idx = timestamps.index(move["timestamp"])
    future_idx = idx + lookahead_periods
    if future_idx >= len(price_history):
        return "unknown"

    future_price = price_history[future_idx]["close"]
    move_direction = 1 if move["pct_change"] > 0 else -1
    future_direction = 1 if future_price > move["to_price"] else -1

    return "continued" if move_direction == future_direction else "reversed"


def score_records(records: list[VerdictRecord], price_history_by_ticker: dict) -> list[VerdictRecord]:
    """Fills in .outcome for each record by checking actual subsequent price action."""
    for r in records:
        history = price_history_by_ticker[r.ticker]
        r.outcome = verify_outcome(history, r.move)
    return records


def compute_metrics(records: list[VerdictRecord]) -> dict:
    """
    Computes the three headline metrics from the README:
      - calibration: for verdicts with confidence >= 0.8, what fraction were "continued"?
      - noise_precision: of verdicts the agent called "noise", what fraction actually
        reversed or were unclear (i.e. the agent was right to be skeptical)?
      - agent_vs_baseline: accuracy of the agent's implied direction vs. the naive
        sentiment-only baseline, on the same move set.
    Verdicts with unknown outcomes (not enough future data) are excluded from scoring.
    """
    scored = [r for r in records if r.outcome in ("continued", "reversed")]
    if not scored:
        return {"error": "No verdicts had enough future data to score. Need more price history."}

    # Calibration
    high_conf = [r for r in scored if r.verdict["confidence"] >= 0.8]
    calibration = (
        sum(1 for r in high_conf if r.outcome == "continued") / len(high_conf)
        if high_conf else None
    )

    # Noise precision
    noise_calls = [r for r in scored if r.verdict["caused_by"] == "noise"]
    noise_precision = (
        sum(1 for r in noise_calls if r.outcome == "reversed") / len(noise_calls)
        if noise_calls else None
    )

    # Agent vs. naive baseline accuracy
    agent_correct = 0
    baseline_correct = 0
    for r in scored:
        actual_direction = "up" if r.move["pct_change"] > 0 else "down"
        # if outcome is "continued", the move direction held; agent implicitly "called"
        # the move's own direction when caused_by != "noise"
        agent_called_direction = actual_direction if r.verdict["caused_by"] != "noise" else "flat"
        if agent_called_direction == actual_direction and r.outcome == "continued":
            agent_correct += 1

        baseline_call = naive_baseline_call(r.evidence)
        if baseline_call == actual_direction and r.outcome == "continued":
            baseline_correct += 1

    return {
        "n_scored": len(scored),
        "calibration_at_0.8_confidence": round(calibration, 3) if calibration is not None else None,
        "noise_call_precision": round(noise_precision, 3) if noise_precision is not None else None,
        "agent_accuracy": round(agent_correct / len(scored), 3),
        "naive_baseline_accuracy": round(baseline_correct / len(scored), 3),
    }
