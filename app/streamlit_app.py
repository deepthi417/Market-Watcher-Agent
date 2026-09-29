import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st
from google.api_core.exceptions import ResourceExhausted

from src.pipeline import process_ticker, run_pipeline, run_backtest, TRACKED_TICKERS

st.set_page_config(page_title="Market-Watcher Agent", layout="wide")
st.title("📈 Market-Watcher Agent")
st.caption("Autonomous causal reasoning over price moves + news, with a backtested eval.")


# --- cached wrappers: same ticker within an hour never re-hits Gemini ---
@st.cache_data(ttl=3600, show_spinner=False)
def cached_process(ticker: str):
    return process_ticker(ticker)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_backtest(tickers: tuple):
    results = run_pipeline(list(tickers))
    return run_backtest(results)


RATE_MSG = ("Gemini API quota reached (rate or daily limit). "
            "Wait a minute and retry, or check your quota in Google AI Studio.")

ticker = st.selectbox("Ticker", TRACKED_TICKERS)

if st.button("Run agent on this ticker"):
    try:
        with st.spinner(f"Analyzing {ticker}..."):
            st.session_state["records"] = (ticker, cached_process(ticker))
    except ResourceExhausted:
        st.warning(RATE_MSG)

if "records" in st.session_state:
    t, records = st.session_state["records"]
    st.subheader(f"Results for {t}")
    if not records:
        st.info("No significant price moves detected in the available window.")
    else:
        for r in records:
            direction = "🟢" if r.move["pct_change"] > 0 else "🔴"
            with st.expander(
                f"{direction} {r.move['timestamp']} · {r.move['pct_change']:+.2f}% · "
                f"verdict: **{r.verdict['caused_by']}** (confidence {r.verdict['confidence']})"
            ):
                st.write("**Explanation:**", r.verdict["explanation"])
                st.write("**Contradicting evidence:**", r.verdict["contradicting_evidence"])
                if r.evidence:
                    st.write("**Evidence used:**")
                    st.dataframe(pd.DataFrame(r.evidence))
                else:
                    st.write("_No news found in the lookback window._")

st.divider()
st.subheader("Backtest evaluation")
st.caption("Compares the agent's causal calls against a naive sentiment-only baseline.")

if st.button("Run full backtest"):
    try:
        with st.spinner("Running pipeline + backtest across all tracked tickers..."):
            st.session_state["metrics"] = cached_backtest(tuple(TRACKED_TICKERS))
    except ResourceExhausted:
        st.warning(RATE_MSG)

metrics = st.session_state.get("metrics")
if metrics:
    if "error" in metrics:
        st.warning(metrics["error"])
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("Agent accuracy", f"{metrics['agent_accuracy']:.0%}")
        col2.metric("Naive baseline accuracy", f"{metrics['naive_baseline_accuracy']:.0%}")
        col3.metric("Noise-call precision",
                    f"{metrics['noise_call_precision']:.0%}" if metrics["noise_call_precision"] is not None else "n/a")
        cal = metrics["calibration_at_0.8_confidence"]
        st.write(f"Calibration @ 0.8+ confidence: {cal}" if cal is not None
                 else "n/a (no high-confidence calls yet)")
        st.write(f"Scored {metrics['n_scored']} verdicts with enough future price data to verify.")
