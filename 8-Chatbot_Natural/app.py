import streamlit as st
import pandas as pd
import numpy as np
import time
import os
import random

from vector_stores import EmbeddingManager, VectorStoreManager
from agent import get_agent_response

# ----------------- STREAMLIT CONFIG & STYLING -----------------
st.set_page_config(
    page_title="Nemotron Natural Agentic RAG",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Glassmorphic UI
st.markdown("""
<style>
    /* Global styles */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Space+Grotesk:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0f0c20 0%, #15102a 50%, #06020f 100%);
        color: #e2e8f0;
    }
    
    /* Title and Header */
    .app-title {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 800;
        background: linear-gradient(90deg, #a855f7 0%, #6366f1 50%, #3b82f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        font-size: 3rem;
        margin-bottom: 0.5rem;
        animation: fadeIn 1.5s ease-out;
    }
    
    .app-subtitle {
        text-align: center;
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Glassmorphic Container */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 20px;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        transition: all 0.3s ease;
    }
    .glass-card:hover {
        border-color: rgba(168, 85, 247, 0.4);
        box-shadow: 0 8px 32px 0 rgba(168, 85, 247, 0.1);
    }
    
    /* Reasoning Container */
    .reasoning-box {
        background: rgba(168, 85, 247, 0.04);
        border-left: 4px solid #a855f7;
        border-radius: 4px 12px 12px 4px;
        padding: 15px;
        margin-bottom: 15px;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.95rem;
        color: #c084fc;
    }
    
    /* Suggestion cards */
    .suggestion-container {
        display: flex;
        flex-wrap: wrap;
        gap: 12px;
        margin-bottom: 25px;
        justify-content: center;
    }
    
    .suggestion-btn {
        background: rgba(99, 102, 241, 0.08);
        border: 1px solid rgba(99, 102, 241, 0.2);
        color: #c7d2fe;
        border-radius: 20px;
        padding: 8px 16px;
        font-size: 0.9rem;
        cursor: pointer;
        transition: all 0.25s ease;
    }
    .suggestion-btn:hover {
        background: linear-gradient(90deg, #6366f1, #a855f7);
        color: white;
        border-color: transparent;
        transform: translateY(-2px);
    }
    
    /* Metrics box */
    .metric-badge {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.85rem;
        margin-right: 10px;
    }
    .metric-latency {
        background: rgba(239, 68, 68, 0.1);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.2);
    }
    .metric-score {
        background: rgba(16, 185, 129, 0.1);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.2);
    }
    .metric-db {
        background: rgba(59, 130, 246, 0.1);
        color: #60a5fa;
        border: 1px solid rgba(59, 130, 246, 0.2);
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
</style>
""", unsafe_allow_html=True)

# ----------------- SESSION STATE & INITIALIZATION -----------------
if "embedding_manager" not in st.session_state:
    st.session_state.embedding_manager = EmbeddingManager()
if "vector_manager" not in st.session_state:
    st.session_state.vector_manager = VectorStoreManager(st.session_state.embedding_manager)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "suggestions" not in st.session_state:
    st.session_state.suggestions = []
if "df_sample" not in st.session_state:
    st.session_state.df_sample = None

# CSV Path
CSV_PATH = "natural.csv"

# Load sample dataset for suggestions & database selection
@st.cache_data
def load_csv_data(path, nrows=5000):
    try:
        return pd.read_csv(path, nrows=nrows)
    except Exception as e:
        st.error(f"Error loading CSV dataset: {e}")
        return None

# Load initial suggestions
if not st.session_state.suggestions and os.path.exists(CSV_PATH):
    df_temp = load_csv_data(CSV_PATH, nrows=50)
    if df_temp is not None:
        st.session_state.suggestions = random.sample(df_temp['query'].tolist(), min(4, len(df_temp)))

# ----------------- SIDEBAR CONFIG -----------------
st.sidebar.markdown("<h2 style='text-align: center; color: #a855f7;'>⚙️ Configurations</h2>", unsafe_allow_html=True)

# Database selection
db_choice = st.sidebar.selectbox(
    "Choose Vector Database",
    ["Chroma DB", "FAISS", "Pinecone"],
    help="Select the vector database engine to use for retrieved context grounding."
)

# Pinecone configuration credentials if needed
pinecone_api_key = ""
pinecone_index_name = "natural-questions"
if db_choice == "Pinecone":
    st.sidebar.info("🔑 Pinecone API Credentials")
    pinecone_api_key = st.sidebar.text_input("Pinecone API Key", value=pinecone_api_key, type="password")
    pinecone_index_name = st.sidebar.text_input("Pinecone Index Name", value=pinecone_index_name)

# Indexing limit slider
index_rows = st.sidebar.slider(
    "Dataset indexing limit",
    min_value=500,
    max_value=20000,
    value=2000,
    step=500,
    help="Number of rows from natural.csv to ingest into the Vector DB. Larger = more knowledge, but longer indexing time."
)

# Index status indicator
is_indexed = False
if db_choice in ["Chroma DB", "FAISS"]:
    is_indexed = st.session_state.vector_manager.load_indexes(db_choice)
elif db_choice == "Pinecone" and pinecone_api_key and pinecone_index_name:
    is_indexed = st.session_state.vector_manager.load_indexes(db_choice, pinecone_api_key, pinecone_index_name)

if is_indexed:
    st.sidebar.success(f"✓ {db_choice} loaded from persistence.")
else:
    st.sidebar.warning(f"⚠️ {db_choice} index not built or empty.")

# Build Index Button
if st.sidebar.button("🔨 Build / Rebuild Index", use_container_width=True):
    with st.sidebar:
        with st.status(f"Ingesting natural.csv into {db_choice}...", expanded=True) as status:
            try:
                # Load CSV
                df = load_csv_data(CSV_PATH, nrows=index_rows)
                if df is None:
                    raise ValueError("Failed to load dataset.")
                
                # Setup database & index
                progress_bar = st.progress(0.0)
                
                def update_prog(current, total):
                    progress_bar.progress(current / total)
                    status.update(label=f"Embedding and Ingesting: {current}/{total} documents...")

                st.session_state.vector_manager.index_data(
                    df, 
                    db_choice, 
                    pinecone_api_key=pinecone_api_key, 
                    pinecone_index_name=pinecone_index_name,
                    progress_callback=update_prog
                )
                
                status.update(label="Indexing completed successfully!", state="complete")
                st.toast(f"Successfully built {db_choice} index with {index_rows} records!", icon="🎉")
                st.rerun()
            except Exception as e:
                status.update(label=f"Error: {e}", state="error")

# Clear chat history button
if st.sidebar.button("🧹 Clear Conversation", use_container_width=True):
    st.session_state.chat_history = []
    st.rerun()

# ----------------- MAIN APP INTERFACE -----------------
st.markdown("<h1 class='app-title'>🧠 Nemotron Agentic RAG</h1>", unsafe_allow_html=True)
st.markdown("<p class='app-subtitle'>An Advanced Agentic RAG QA Chatbot utilizing Chroma, FAISS, Pinecone, and Nvidia Nemotron-3 Super 120B</p>", unsafe_allow_html=True)

# Question Suggestion Cards
st.write("💡 **Try asking one of these questions from the dataset:**")
cols = st.columns(len(st.session_state.suggestions))
selected_suggestion = None

for idx, sugg in enumerate(st.session_state.suggestions):
    with cols[idx]:
        if st.button(sugg, key=f"sugg_{idx}", use_container_width=True):
            selected_suggestion = sugg

# Chat Input & history
user_query = st.chat_input("Ask any natural question...")

# Handle suggestion selection
if selected_suggestion:
    user_query = selected_suggestion

# Display Chat History
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        # Show metrics if available
        if "metrics" in message:
            m = message["metrics"]
            st.markdown(
                f"<span class='metric-badge metric-db'>🗄️ {m['db']}</span>"
                f"<span class='metric-badge metric-latency'>⚡ {m['latency']:.2f}s</span>"
                f"<span class='metric-badge metric-score'>🎯 Conf: {m['score']:.2f}</span>",
                unsafe_allow_html=True
            )

# Execute RAG flow
if user_query:
    # Display user question
    with st.chat_message("user"):
        st.write(user_query)
        
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    
    # RAG Retrieval
    t0 = time.time()
    with st.spinner("Retrieving relevant context from Vector DB..."):
        retrieved_docs = st.session_state.vector_manager.query(
            user_query, 
            db_choice, 
            top_k=3,
            pinecone_api_key=pinecone_api_key,
            pinecone_index_name=pinecone_index_name
        )
    t_retrieval = time.time() - t0
    
    # Calculate best similarity score
    best_score = retrieved_docs[0]["score"] if retrieved_docs else 0.0
    
    # Display system response placeholder
    with st.chat_message("assistant"):
        # Setup reasoning expander
        reasoning_expander = st.expander("💭 Agentic Reasoning Process", expanded=True)
        reasoning_placeholder = reasoning_expander.empty()
        
        # Setup final answer placeholder
        answer_placeholder = st.empty()
        
        reasoning_text = ""
        answer_text = ""
        
        # Stream response
        t1 = time.time()
        for response_type, chunk in get_agent_response(user_query, retrieved_docs, st.session_state.chat_history[:-1]):
            if response_type == "reasoning":
                reasoning_text += chunk
                reasoning_placeholder.markdown(f"<div class='reasoning-box'>{reasoning_text}</div>", unsafe_allow_html=True)
            elif response_type == "content":
                answer_text += chunk
                answer_placeholder.write(answer_text)
            elif response_type == "error":
                st.error(chunk)
                answer_text = f"An error occurred: {chunk}"
                answer_placeholder.write(answer_text)
                
        t_total = time.time() - t0
        
        # Render metrics badges
        metrics = {
            "db": db_choice,
            "latency": t_total,
            "score": best_score
        }
        
        st.markdown(
            f"<span class='metric-badge metric-db'>🗄️ {metrics['db']}</span>"
            f"<span class='metric-badge metric-latency'>⚡ {metrics['latency']:.2f}s</span>"
            f"<span class='metric-badge metric-score'>🎯 Conf: {metrics['score']:.2f}</span>",
            unsafe_allow_html=True
        )
        
        # Append to history with metrics
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer_text,
            "metrics": metrics
        })
        
        # Auto-scroll or force refresh to reset suggestion click state
        if selected_suggestion:
            st.rerun()
