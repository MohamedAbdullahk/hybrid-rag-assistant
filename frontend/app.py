import os
import time
import requests
import streamlit as st

# Environment-aware API Configurations
# Explicitly default to the container_name used in docker-compose.yml
DEFAULT_BACKEND = "http://hybrid_rag_backend:8000"
BACKEND_HOST = os.getenv("BACKEND_HOST") or os.getenv("BACKEND_URL") or DEFAULT_BACKEND

DOC_API_URL = f"{BACKEND_HOST}/api/v1/document"
CHAT_API_URL = f"{BACKEND_HOST}/api/v1/chat"

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
            st.rerun()

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