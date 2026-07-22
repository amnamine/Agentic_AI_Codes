import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import time
from datetime import datetime
import plotly.express as px

# Import our agent engine
import agent_engine

# --- Page Config ---
st.set_page_config(
    page_title="Antigravity Telecom Agentic RAG Hub",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Inject Premium CSS (Dark Theme, Glassmorphism, Animations) ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Space+Grotesk:wght@400;600&display=swap');

/* Global Styles */
html, body, [data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at top right, #0d1117 0%, #07090e 100%);
    font-family: 'Outfit', sans-serif;
    color: #e2e8f0;
}

/* Titles and Headers */
h1, h2, h3 {
    font-family: 'Space Grotesk', sans-serif;
    background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
}

/* Glassmorphism Cards */
.glass-card {
    background: rgba(15, 23, 42, 0.45);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 20px;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.glass-card:hover {
    border-color: rgba(0, 242, 254, 0.4);
    box-shadow: 0 8px 32px 0 rgba(0, 242, 254, 0.15);
}

/* Sidebar Custom Styling */
[data-testid="stSidebar"] {
    background-color: rgba(6, 9, 15, 0.95);
    border-right: 1px solid rgba(255, 255, 255, 0.05);
}

/* Metrics Styling */
[data-testid="stMetricValue"] {
    font-size: 2rem;
    font-weight: 800;
    color: #00f2fe !important;
}

/* Glowing Border for special items */
.glow-border {
    border: 1px solid transparent;
    background-image: linear-gradient(rgba(15, 23, 42, 0.8), rgba(15, 23, 42, 0.8)), 
                      linear-gradient(135deg, #00f2fe 0%, #4facfe 100%);
    background-origin: border-box;
    background-clip: content-box, border-box;
    border-radius: 16px;
}

/* Custom Chat Bubbles */
.chat-bubble {
    padding: 14px 18px;
    border-radius: 18px;
    margin-bottom: 12px;
    max-width: 80%;
    line-height: 1.5;
}

.chat-user {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid rgba(255, 255, 255, 0.08);
    align-self: flex-end;
    margin-left: auto;
    border-bottom-right-radius: 4px;
}

.chat-assistant {
    background: linear-gradient(135deg, #1e1b4b 0%, #0f0b29 100%);
    border: 1px solid rgba(99, 102, 241, 0.2);
    align-self: flex-start;
    border-bottom-left-radius: 4px;
    box-shadow: 0 4px 12px rgba(99, 102, 241, 0.1);
}

/* Thinking Indicator Log */
.thought-log-container {
    background: rgba(30, 41, 59, 0.2);
    border-left: 3px solid #6366f1;
    padding: 10px 15px;
    margin: 8px 0;
    border-radius: 0 8px 8px 0;
    font-family: 'Space Grotesk', monospace;
    font-size: 0.9rem;
}

/* Animations */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
}

.animate-fade {
    animation: fadeIn 0.4s ease-out forwards;
}
</style>
""", unsafe_allow_html=True)

# --- Initialize session states ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am your Antigravity Advanced Telecom Copilot. I can query our knowledge base of 230k+ records, simulate account bill audits, cancel/modify contracts, or raise engineering tickets. How can I assist you today?"}
    ]
if "thought_history" not in st.session_state:
    st.session_state.thought_history = []
if "audits_log" not in st.session_state:
    st.session_state.audits_log = []
if "tickets_log" not in st.session_state:
    st.session_state.tickets_log = []

# --- Cached Data Loading for Dashboard Analytics ---
@st.cache_data
def load_dataset_stats():
    csv_path = "bitext-telco-llm-chatbot-training-dataset.csv"
    if os.path.exists(csv_path):
        # Read small sample or optimize to read specific columns for memory efficiency
        df = pd.read_csv(csv_path, usecols=["category", "intent"])
        cat_counts = df["category"].value_counts().reset_index()
        cat_counts.columns = ["Category", "Count"]
        
        intent_counts = df["intent"].value_counts().head(8).reset_index()
        intent_counts.columns = ["Intent", "Count"]
        
        total_rows = len(df)
        return total_rows, cat_counts, intent_counts
    return 232002, None, None

total_rows, cat_counts, intent_counts = load_dataset_stats()

# --- Sidebar Content ---
with st.sidebar:
    st.markdown("<h2 style='text-align: center;'>⚡ SYSTEM METRICS</h2>", unsafe_allow_html=True)
    
    # Glowing Server Status Card
    st.markdown("""
    <div class='glass-card glow-border' style='padding: 15px;'>
        <div style='display: flex; align-items: center; gap: 10px;'>
            <span style='height: 12px; width: 12px; background-color: #00ff66; border-radius: 50%; display: inline-block; box-shadow: 0 0 10px #00ff66;'></span>
            <strong style='color: #e2e8f0;'>AI Core: llama-3.3-70b-versatile</strong>
        </div>
        <div style='font-size: 0.85rem; color: #94a3b8; margin-top: 5px;'>
            Connection: Groq Cloud API (Active)<br>
            Vector Engine: FAISS L2 Flat Index
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Database stats
    st.metric(label="RAG Knowledge Base Size", value=f"{total_rows:,} rows")
    
    # Quick Simulation Profile Selector
    st.markdown("### 👤 Customer Profile Presets")
    profile = st.selectbox(
        "Choose a template scenario:",
        ["None", "Dispute charges on recent invoice", "Internet connection is down", "Request support ticket for router replacement", "Cancel subscription package"]
    )
    
    if profile != "None" and st.button("Load Profile Scenario"):
        st.session_state.messages.append({"role": "user", "content": profile})
        st.rerun()

    # Dynamic Chatbot Personality Selector
    st.markdown("### 🎭 Response Personality")
    chatbot_tone = st.selectbox(
        "Select Agent Tone:",
        ["Empathetic Support", "Strict & Technical", "Sales & Upbeat", "Concise & Direct"]
    )
        
    st.markdown("### ⚙️ Utilities")
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I am your Antigravity Advanced Telecom Copilot. I can query our knowledge base of 230k+ records, simulate account bill audits, cancel/modify contracts, or raise engineering tickets. How can I assist you today?"}
        ]
        st.session_state.thought_history = []
        st.rerun()

    st.markdown("---")
    st.markdown("<p style='text-align: center; color: #64748b; font-size: 0.8rem;'>Antigravity Telecom Chatbot v1.0.0</p>", unsafe_allow_html=True)


# --- Main Dashboard Tabs ---
tab_chat, tab_analytics, tab_vector_db, tab_logs = st.tabs([
    "💬 Agentic RAG Chatbot",
    "📊 Dataset Operations Analytics",
    "🔍 Knowledge Vector DB Explorer",
    "⚙️ Agent Audits & Ticketing Logs"
])

# --- TAB 1: Chat interface ---
with tab_chat:
    st.markdown("<h1 style='margin-bottom: 5px;'>⚡ ANTIGRAVITY TELECOM ASSISTANT</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94a3b8;'>Advanced Agentic AI Orchestrator running on Groq Llama 3.3 70B & Vector DB Retrieval</p>", unsafe_allow_html=True)
    
    col_chat_main, col_thoughts = st.columns([5, 3])
    
    with col_chat_main:
        # Chat container
        chat_placeholder = st.container(height=500)
        
        with chat_placeholder:
            for message in st.session_state.messages:
                bubble_class = "chat-user" if message["role"] == "user" else "chat-assistant"
                st.markdown(
                    f"<div class='chat-bubble {bubble_class} animate-fade'>{message['content']}</div>",
                    unsafe_allow_html=True
                )
                
        # Suggestion Chips
        st.markdown("<p style='font-size: 0.9rem; color: #94a3b8; margin-bottom: 5px; margin-top: 10px;'>💡 Quick Suggestions:</p>", unsafe_allow_html=True)
        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        triggered_prompt = None
        if col_s1.button("🔍 Check Network Status", use_container_width=True):
            triggered_prompt = "Is there a network outage in my area or is my internet connection down?"
        if col_s2.button("💳 Dispute Bill Charge", use_container_width=True):
            triggered_prompt = "Audit my account ACC-5012 for a disputed charge of $85 on my bill"
        if col_s3.button("📦 Upgrade Mobile Plan", use_container_width=True):
            triggered_prompt = "I want to upgrade my mobile data package to a higher speed plan"
        if col_s4.button("🎟️ Replace Faulty Router", use_container_width=True):
            triggered_prompt = "Request support ticket for router replacement because my router is blinking red"

        # Chat input handling
        prompt = None
        chat_input_val = st.chat_input("Enter your telecom inquiry... (e.g. 'Audit my account ACC-4912 for $120 dispute')")
        if chat_input_val:
            prompt = chat_input_val
        elif triggered_prompt:
            prompt = triggered_prompt

        if prompt:
            # Append user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Run Agent
            with st.spinner("Agent is planning reasoning steps..."):
                response, thoughts = agent_engine.run_agentic_rag(
                    prompt, 
                    st.session_state.messages[:-1],
                    tone=chatbot_tone
                )
                
            # Log thoughts
            st.session_state.thought_history.append({
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "query": prompt,
                "logs": thoughts
            })
            
            # Catch simulated events and log them to respective lists
            # Billing simulator audit logs
            if "Audit ID:" in "".join(thoughts):
                st.session_state.audits_log.append({
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "query": prompt,
                    "summary": next((t for t in thoughts if "Audit ID" in t), "Audit completed")
                })
            # Ticket creator logs
            if "Ticket ID:" in "".join(thoughts):
                st.session_state.tickets_log.append({
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "query": prompt,
                    "summary": next((t for t in thoughts if "Ticket ID" in t), "Ticket registered")
                })
                
            # Append assistant response
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()

    with col_thoughts:
        st.markdown("### ⚙️ Live Agent Reasoning Chain")
        if st.session_state.thought_history:
            latest = st.session_state.thought_history[-1]
            st.markdown(f"**Last Query:** *{latest['query']}*")
            for log in latest['logs']:
                st.markdown(f"<div class='thought-log-container'>{log}</div>", unsafe_allow_html=True)
        else:
            st.markdown(
                "<div style='color: #64748b; font-style: italic; border: 1px dashed rgba(255,255,255,0.1); padding: 20px; border-radius: 8px; text-align: center;'>"
                "Awaiting user inquiry to track agent's reasoning..."
                "</div>",
                unsafe_allow_html=True
            )

# --- TAB 2: Dataset Analytics ---
with tab_analytics:
    st.markdown("## 📊 Dataset Distribution & Categorization Insights")
    st.markdown("This live dashboard visualizes the distribution of intents and categories in the **232,002 record** Bitext Telco dataset, showing the taxonomy the RAG model understands.")
    
    if cat_counts is not None and intent_counts is not None:
        col_c1, col_c2 = st.columns(2)
        
        with col_c1:
            st.markdown("### 📂 Query Categories")
            fig1 = px.bar(
                cat_counts, 
                x="Count", 
                y="Category", 
                orientation="h",
                color="Count",
                color_continuous_scale="Viridis",
                template="plotly_dark"
            )
            fig1.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig1, use_container_width=True)
            
        with col_c2:
            st.markdown("### 🎯 Top 8 Query Intents")
            fig2 = px.pie(
                intent_counts, 
                values="Count", 
                names="Intent",
                hole=0.4,
                color_discrete_sequence=px.colors.sequential.Bluyl_r,
                template="plotly_dark"
            )
            fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Dataset statistics are currently loading or unavailable.")

# --- TAB 3: Vector DB Explorer ---
with tab_vector_db:
    st.markdown("## 🔍 Knowledge Base & Semantic Search Explorer")
    st.markdown("Test query retrieval directly from the FAISS database index below. Retrieve raw instruction sets, intents, and templates.")
    
    search_q = st.text_input("Enter semantic search query (e.g. 'internet speed is slow'):", "how to cancel line")
    top_n = st.slider("Number of matches (K):", 1, 5, 3)
    
    if st.button("Search FAISS DB"):
        with st.spinner("Querying vector index..."):
            retrieval_results = agent_engine.retrieve_telecom_policy(search_q, top_k=top_n)
            
        for i, item in enumerate(retrieval_results):
            if "error" in item:
                st.error(item["error"])
                break
            st.markdown(f"""
            <div class='glass-card'>
                <h4>Match #{i+1} | Semantic Confidence: <span style='color: #00f2fe;'>{item['confidence']:.2%}</span></h4>
                <p><strong>Database Instruction Phrasing:</strong> "{item['instruction']}"</p>
                <p><strong>Intent Category:</strong> <span style='background-color: #1e1b4b; padding: 3px 8px; border-radius: 4px;'>{item['intent']}</span> | <strong>Tags:</strong> <code>{item['tags']}</code></p>
                <blockquote style='border-left: 4px solid #00f2fe; padding-left: 10px; margin-left: 0; color: #cbd5e1;'>
                    <strong>Template Response:</strong><br>{item['response']}
                </blockquote>
            </div>
            """, unsafe_allow_html=True)

# --- TAB 4: Audits & Ticketing Logs ---
with tab_logs:
    st.markdown("## ⚙️ Automated Actions and Integrations Registry")
    st.markdown("This panel logs tool calls executed by the agent, showing actions like simulated billing adjustments or technical support ticket updates.")
    
    col_l1, col_l2 = st.columns(2)
    
    with col_l1:
        st.markdown("### 💳 Simulated Bill Dispute Audits")
        if st.session_state.audits_log:
            for audit in st.session_state.audits_log:
                st.markdown(f"""
                <div style='background-color: rgba(30, 41, 59, 0.3); border: 1px solid rgba(0, 242, 254, 0.2); padding: 12px; border-radius: 8px; margin-bottom: 10px;'>
                    <span style='font-size: 0.8rem; color: #94a3b8;'>{audit['timestamp']}</span><br>
                    <strong>Dispute Prompt:</strong> <em>{audit['query']}</em><br>
                    <span style='color: #00f2fe;'>{audit['summary']}</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No billing disputes simulated yet.")
            
    with col_l2:
        st.markdown("### 🎟️ Technical & Engineering Tickets")
        if st.session_state.tickets_log:
            for ticket in st.session_state.tickets_log:
                st.markdown(f"""
                <div style='background-color: rgba(30, 41, 59, 0.3); border: 1px solid rgba(99, 102, 241, 0.2); padding: 12px; border-radius: 8px; margin-bottom: 10px;'>
                    <span style='font-size: 0.8rem; color: #94a3b8;'>{ticket['timestamp']}</span><br>
                    <strong>Issue Prompt:</strong> <em>{ticket['query']}</em><br>
                    <span style='color: #818cf8;'>{ticket['summary']}</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No support tickets created yet.")
