import os
import time
import requests
import streamlit as st
import pandas as pd
import altair as alt

# Environment-aware API Configurations
# Explicitly default to the container_name used in docker-compose.yml
DEFAULT_BACKEND = "http://hybrid_rag_backend:8000"
BACKEND_HOST = os.getenv("BACKEND_HOST") or os.getenv("BACKEND_URL") or DEFAULT_BACKEND

DOC_API_URL = f"{BACKEND_HOST}/api/v1/document"
CHAT_API_URL = f"{BACKEND_HOST}/api/v1/chat"
EVAL_API_URL = f"{BACKEND_HOST}/api/v1/eval/compare"

st.set_page_config(
    page_title="Conversational Hybrid RAG Assistant",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS for latency badge layout
st.markdown("""
<style>
.latency-badge {
    font-size: 0.75rem;
    color: #6c757d;
    text-align: right;
    margin-top: 4px;
}
</style>
""", unsafe_allow_html=True)

# Initialize Session State Variables
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "is_processed" not in st.session_state:
    st.session_state.is_processed = False
if "processing_time" not in st.session_state:
    st.session_state.processing_time = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "eval_result" not in st.session_state:
    st.session_state.eval_result = None

st.title("🤖 Conversational Hybrid RAG Assistant")

# --- SIDEBAR: Document Upload section ---
with st.sidebar:
    st.header("📄 Document Upload")

    uploaded_file = st.file_uploader(
        "Upload a PDF for this session",
        type=["pdf"],
        disabled=st.session_state.is_processed
    )

    if uploaded_file is not None and not st.session_state.is_processed:
        if st.button("Process Document", type="primary"):
            with st.spinner("Uploading document to backend..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                    res = requests.post(f"{DOC_API_URL}/upload", files=files)

                    if res.status_code == 200:
                        data = res.json()
                        session_id = data["session_id"]
                        st.session_state.session_id = session_id

                        # Poll status endpoint
                        status_box = st.empty()
                        status_box.info("Indexing vector chunks & Knowledge Graph...")

                        completed = False
                        while not completed:
                            time.sleep(1)
                            status_res = requests.get(f"{DOC_API_URL}/status/{session_id}")
                            if status_res.status_code == 200:
                                job_data = status_res.json()
                                if job_data.get("status") == "completed":
                                    completed = True
                                    elapsed = job_data.get("elapsed_time", 0.0)
                                    st.session_state.is_processed = True
                                    st.session_state.processing_time = elapsed
                                    status_box.success(f"Document processed in {elapsed}s. You can start chatting!")
                                elif job_data.get("status") == "failed":
                                    status_box.error(f"Failed: {job_data.get('error')}")
                                    break
                    else:
                        st.error(f"Upload error: {res.text}")
                except Exception as e:
                    st.error(f"Connection error: {str(e)}")

    if st.session_state.is_processed:
        st.success(f"Active Session: `{st.session_state.session_id[:8]}...`")
        st.info(f"Indexing Time: `{st.session_state.processing_time}s`")

        if st.button("Clear Session / Upload New PDF"):
            st.session_state.session_id = None
            st.session_state.is_processed = False
            st.session_state.processing_time = None
            st.session_state.messages = []
            st.session_state.eval_result = None
            st.rerun()

    # --- Strategy Evaluation Form ---
    if st.session_state.is_processed:
        st.divider()
        st.header("📊 Evaluate Strategies")

        with st.form("eval_form"):
            eval_question = st.text_area(
                "Test question",
                placeholder="e.g. Which service owns the playback_queue table?"
            )
            eval_expected = st.text_area(
                "Expected answer (optional)",
                placeholder="e.g. Playback & Session Service"
            )
            run_eval = st.form_submit_button("Run Evaluation", type="primary")

        if run_eval:
            if not eval_question.strip():
                st.warning("Please enter a test question.")
            else:
                with st.spinner("Running all 4 strategies... (~15-30s)"):
                    try:
                        payload = {
                            "session_id": st.session_state.session_id,
                            "question": eval_question,
                            "expected_answer": eval_expected
                        }
                        res = requests.post(EVAL_API_URL, json=payload, timeout=180)
                        if res.status_code == 200:
                            st.session_state.eval_result = res.json()
                            st.success("Evaluation completed! See results on the right.")
                        else:
                            st.error(f"Evaluation error: {res.text}")
                    except Exception as e:
                        st.error(f"Connection failed: {str(e)}")


# --- EVALUATION RESULTS PANEL ---
if st.session_state.eval_result:
    ev = st.session_state.eval_result
    results = ev["results"]
    best_name = ev["best_strategy"]
    best = next(r for r in results if r["strategy"] == best_name)

    with st.expander("📊 Strategy Evaluation Results", expanded=True):
        st.markdown(f"**Question:** {ev['query']}")

        # Summary cards
        c1, c2, c3 = st.columns(3)
        c1.metric("🏆 Best Strategy", best_name)
        c2.metric("Best Overall Score", f"{best['overall_score']:.2f}")
        c3.metric("Pass Threshold", f"{ev['threshold']:.2f}")

        # Honest note when the top score is tied
        top_score = best["overall_score"]
        tied = [r["strategy"] for r in results if r["overall_score"] == top_score]
        if len(tied) > 1:
            st.info(
                f"{len(tied)} strategies tied at {top_score:.2f} ({', '.join(tied)}) — "
                f"'{best_name}' was picked only because it was fastest."
            )

        # Score table
        table_rows = [
            {
                "Strategy": ("🏆 " if r["strategy"] == best_name else "") + r["strategy"],
                "Faithfulness": r["faithfulness_score"],
                "Answer Relevancy": r["relevancy_score"],
                "Context Relevancy": r["context_relevancy_score"],
                "Overall": r["overall_score"],
                "Latency (s)": r["latency_seconds"],
                "Result": "✅ PASS" if r["passed"] else "❌ FAIL",
            }
            for r in results
        ]
        st.dataframe(table_rows, use_container_width=True, hide_index=True)

        # Bar chart: 3 metrics per strategy
        chart_df = pd.DataFrame(
            {
                "Faithfulness": [r["faithfulness_score"] for r in results],
                "Answer Relevancy": [r["relevancy_score"] for r in results],
                "Context Relevancy": [r["context_relevancy_score"] for r in results],
            },
            index=[r["strategy"] for r in results],
        )
        chart_long = (
            chart_df.reset_index()
            .melt(id_vars="index", var_name="Metric", value_name="Score")
            .rename(columns={"index": "Strategy"})
        )
        chart = alt.Chart(chart_long).mark_bar().encode(
            x=alt.X("Strategy:N", title=None, axis=alt.Axis(labelAngle=0)),
            xOffset="Metric:N",
            y=alt.Y("Score:Q", scale=alt.Scale(domain=[0, 1])),
            color="Metric:N",
            tooltip=["Strategy", "Metric", "Score"],
        )
        st.altair_chart(chart, use_container_width=True)

        # Per-strategy answer + judge reason
        tabs = st.tabs([r["strategy"] for r in results])
        for tab, r in zip(tabs, results):
            with tab:
                st.markdown(f"**Answer:** {r['actual_output']}")
                st.caption(f"🧑‍⚖️ Judge reason: {r['reason']}")


# --- MAIN CHAT INTERFACE ---

# Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg["role"] == "assistant":
            # Display source citations if present
            if msg.get("sources"):
                st.caption(f"📚 **Sources:** {', '.join(msg['sources'])}")

            # Display response time at bottom-right
            if msg.get("timings"):
                total_time = msg["timings"].get("total", 0.0)
                st.markdown(f'<div class="latency-badge">⏱️ {total_time:.3f}s</div>', unsafe_allow_html=True)

                # Expandable timing breakdown
                with st.expander("📊 Latency Breakdown"):
                    for stage, duration in msg["timings"].items():
                        st.write(f"- **{stage.capitalize()}**: {duration}s")

# Chat Input Widget (Disabled until document is processed)
chat_disabled = not st.session_state.is_processed
input_hint = "Upload and process a PDF to enable chat" if chat_disabled else "Ask a question about your document..."

if user_input := st.chat_input(input_hint, disabled=chat_disabled):
    # Add User message to state and UI
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Process AI Response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                payload = {
                    "session_id": st.session_state.session_id,
                    "message": user_input
                }
                response = requests.post(CHAT_API_URL, json=payload)

                if response.status_code == 200:
                    res_data = response.json()
                    answer = res_data["answer"]
                    sources = res_data.get("sources", [])
                    timings = res_data.get("timings", {})

                    st.markdown(answer)

                    if sources:
                        st.caption(f"📚 **Sources:** {', '.join(sources)}")

                    total_time = timings.get("total", 0.0)
                    st.markdown(f'<div class="latency-badge">⏱️ {total_time:.3f}s</div>', unsafe_allow_html=True)

                    with st.expander("📊 Latency Breakdown"):
                        for stage, duration in timings.items():
                            st.write(f"- **{stage.capitalize()}**: {duration}s")

                    # Save Assistant message to state
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                        "timings": timings
                    })
                else:
                    st.error(f"Error from server: {response.text}")
            except Exception as e:
                st.error(f"Connection failed: {str(e)}")