import streamlit as st
import pandas as pd
import json
import sqlite3
import os
import plotly.express as px
from datetime import datetime

# Import agent and db manager functions
from db_manager import (
    get_db_connection,
    list_all_tickets,
    create_ticket,
    update_ticket,
    FAISS_INDEX_PATH,
    METADATA_PATH,
    VectorSearcher
)
from agent import run_agent_loop, API_KEY, MODEL_NAME

# Page Config
st.set_page_config(
    page_title="Helix AI - Autonomous Ticketing Agent",
    page_icon="🎫",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (CSS)
st.markdown("""
<style>
    /* Google Fonts import */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

    /* Variables and Theme Settings */
    :root {
        --bg-gradient: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        --card-bg: rgba(30, 41, 59, 0.7);
        --accent-purple: #8b5cf6;
        --accent-blue: #3b82f6;
        --border-color: rgba(255, 255, 255, 0.1);
        --text-main: #f8fafc;
        --text-muted: #94a3b8;
    }

    /* Font Styles */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }
    
    h1, h2, h3 {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 700;
        letter-spacing: -0.02em;
    }

    /* Main Container Glassmorphism */
    .stApp {
        background: linear-gradient(135deg, #080710 0%, #0f172a 100%) !important;
        color: var(--text-main) !important;
    }

    /* Custom Title Style */
    .main-title {
        font-size: 2.8rem;
        background: linear-gradient(45deg, #8b5cf6, #3b82f6, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 5px;
        font-weight: 800;
        animation: fadeIn 1.2s ease-out;
    }
    
    .subtitle {
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 25px;
    }

    /* Metric Cards Styling */
    .metric-card {
        background: rgba(30, 41, 59, 0.45);
        border: 1px solid var(--border-color);
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.15);
        backdrop-filter: blur(10px);
        transition: transform 0.3s ease, border-color 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        border-color: var(--accent-purple);
    }
    
    .metric-val {
        font-size: 2.2rem;
        font-weight: 700;
        color: #f8fafc;
        line-height: 1;
        margin-top: 8px;
    }

    /* Status Badges */
    .badge {
        padding: 4px 8px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-open { background-color: rgba(59, 130, 246, 0.2); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.3); }
    .badge-progress { background-color: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.3); }
    .badge-resolved { background-color: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); }

    /* Custom Chat Message styles */
    .chat-bubble {
        padding: 15px 20px;
        border-radius: 18px;
        margin-bottom: 12px;
        max-width: 80%;
        line-height: 1.5;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    }
    .chat-user {
        background: linear-gradient(135deg, #4f46e5 0%, #3b82f6 100%);
        color: white;
        margin-left: auto;
        border-bottom-right-radius: 4px;
    }
    .chat-agent {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid var(--border-color);
        color: #f1f5f9;
        margin-right: auto;
        border-bottom-left-radius: 4px;
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
</style>
""", unsafe_allow_html=True)

# Helper function to get clean styling for priority
def get_priority_color(p):
    if p.lower() == "high":
        return "🔴 High"
    elif p.lower() == "medium":
        return "🟡 Medium"
    return "🟢 Low"

# Main Application Layout
st.markdown("<h1 class='main-title'>🎫 HELIX AI</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Autonomous Customer Support & Ticketing Agent Engine</p>", unsafe_allow_html=True)

# Sidebar Configuration
st.sidebar.markdown("""
<div style='background: rgba(30, 41, 59, 0.45); padding: 15px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.1); margin-bottom: 20px;'>
    <h4 style='margin-top: 0; color: #f8fafc;'>⚙️ Settings Panel</h4>
</div>
""", unsafe_allow_html=True)
api_key_input = st.sidebar.text_input(
    "Google Gemini API Key",
    value=API_KEY,
    type="password",
    help="Default key is pre-configured. Paste your own Google AI Studio API key here if needed."
)

# Navigation tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "💬 AI Agent Chatbot", 
    "📊 Real-time Dashboard", 
    "🗄️ Database Console", 
    "📖 RAG Knowledge Base"
])

# Initialize session state for chatbot
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "agent_logs" not in st.session_state:
    st.session_state.agent_logs = []
if "temp_prompt" not in st.session_state:
    st.session_state.temp_prompt = None

# TAB 1: Chatbot Interface
with tab1:
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.subheader("Interactive Support Agent")
        
        # Chat history container
        chat_container = st.container(height=420)
        with chat_container:
            for message in st.session_state.chat_history:
                if message["role"] == "user":
                    st.markdown(f'<div class="chat-bubble chat-user">{message["content"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="chat-bubble chat-agent">{message["content"]}</div>', unsafe_allow_html=True)
                    
        # Suggested Prompts
        st.markdown("<p style='font-size: 0.9rem; font-weight: 600; color: #94a3b8; margin-bottom: 5px;'>💡 Suggestion Shortcuts:</p>", unsafe_allow_html=True)
        s_col1, s_col2, s_col3 = st.columns(3)
        with s_col1:
            if st.button("🎫 How do I cancel my ticket?", use_container_width=True):
                st.session_state.temp_prompt = "How do I cancel my ticket for the sports match?"
        with s_col2:
            if st.button("➕ Open Support Ticket", use_container_width=True):
                st.session_state.temp_prompt = "Can you open a support ticket for my login issue? Name: Sam Lee, email: sam.l@example.com. Description: Whenever I try to log in, it says account suspended."
        with s_col3:
            if st.button("🔍 Check ticket TKT-1002", use_container_width=True):
                st.session_state.temp_prompt = "Show me the details and current status of ticket TKT-1002"

        # User input box
        prompt = st.chat_input("How can I assist you with your ticket today?")
        
        # Determine if we should process
        active_prompt = None
        if prompt:
            active_prompt = prompt
        elif st.session_state.temp_prompt:
            active_prompt = st.session_state.temp_prompt
            st.session_state.temp_prompt = None
            
        if active_prompt:
            # Display user message instantly
            st.session_state.chat_history.append({"role": "user", "content": active_prompt})
            
            # Show processing state
            with chat_container:
                st.markdown(f'<div class="chat-bubble chat-user">{active_prompt}</div>', unsafe_allow_html=True)
                
                with st.spinner("Helix Agent thinking..."):
                    # Execute agent loop generator
                    agent_gen = run_agent_loop(active_prompt, chat_history=st.session_state.chat_history[:-1], custom_api_key=api_key_input)
                    
                    final_response = ""
                    log_updates = []
                    
                    # Consume generator to display tool updates or final response
                    for response_text, tool_logs in agent_gen:
                        if tool_logs:
                            log_updates.extend(tool_logs)
                            for log in tool_logs:
                                st.info(log)
                        if response_text:
                            final_response = response_text
                    
                    if final_response:
                        st.session_state.chat_history.append({"role": "assistant", "content": final_response})
                        st.markdown(f'<div class="chat-bubble chat-agent">{final_response}</div>', unsafe_allow_html=True)
                        if log_updates:
                            st.session_state.agent_logs.append({
                                "timestamp": datetime.now().strftime("%H:%M:%S"),
                                "prompt": active_prompt,
                                "logs": log_updates
                            })
            st.rerun()

    with col_right:
        st.subheader("Agent Operations Log")
        st.write("Real-time background execution step tracing.")
        
        if not st.session_state.agent_logs:
            st.info("No tool executions recorded yet. Talk to the agent to trigger database operations or FAISS queries.")
        else:
            for idx, entry in enumerate(reversed(st.session_state.agent_logs)):
                with st.expander(f"🔄 Prompt: \"{entry['prompt'][:30]}...\" at {entry['timestamp']}", expanded=(idx==0)):
                    for log in entry['logs']:
                        st.code(log, language="bash")

# TAB 2: Real-time Dashboard
with tab2:
    st.subheader("Ticketing Analytics Dashboard")
    tickets = list_all_tickets()
    
    if not tickets:
        st.warning("No tickets found to display dashboard metrics.")
    else:
        df_tickets = pd.DataFrame(tickets)
        
        # Calculate Metrics
        total_t = len(df_tickets)
        open_t = len(df_tickets[df_tickets["status"] == "Open"])
        prog_t = len(df_tickets[df_tickets["status"] == "In Progress"])
        res_t = len(df_tickets[df_tickets["status"] == "Resolved"])
        res_rate = (res_t / total_t * 100) if total_t > 0 else 0
        
        # Metric Columns Row 1
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        with m_col1:
            st.markdown(f"""
            <div class="metric-card">
                <div style="color: var(--text-muted); font-size: 0.9rem;">Total Tickets</div>
                <div class="metric-val">{total_t}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with m_col2:
            st.markdown(f"""
            <div class="metric-card">
                <div style="color: #60a5fa; font-size: 0.9rem;">🟢 Open Tickets</div>
                <div class="metric-val">{open_t}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with m_col3:
            st.markdown(f"""
            <div class="metric-card">
                <div style="color: #fbbf24; font-size: 0.9rem;">🟡 In Progress</div>
                <div class="metric-val">{prog_t}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with m_col4:
            st.markdown(f"""
            <div class="metric-card">
                <div style="color: #34d399; font-size: 0.9rem;">✅ Resolved</div>
                <div class="metric-val">{res_t}</div>
            </div>
            """, unsafe_allow_html=True)
            
        # Metric Columns Row 2 (KPIs)
        st.write("")
        k_col1, k_col2, k_col3 = st.columns(3)
        with k_col1:
            st.markdown(f"""
            <div class="metric-card">
                <div style="color: #a78bfa; font-size: 0.9rem;">📈 Resolution Rate</div>
                <div class="metric-val">{res_rate:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        with k_col2:
            # Mock average resolution time
            st.markdown(f"""
            <div class="metric-card">
                <div style="color: #f472b6; font-size: 0.9rem;">⚡ Avg Resolution Time</div>
                <div class="metric-val">2.4h</div>
            </div>
            """, unsafe_allow_html=True)
        with k_col3:
            # RAG Knowledge base capacity
            import pickle
            try:
                with open(METADATA_PATH, "rb") as f:
                    kb_size = len(pickle.load(f))
            except:
                kb_size = 3000
            st.markdown(f"""
            <div class="metric-card">
                <div style="color: #38bdf8; font-size: 0.9rem;">📖 FAQ Knowledge Pool</div>
                <div class="metric-val">{kb_size} items</div>
            </div>
            """, unsafe_allow_html=True)

        st.write("")
        st.write("")
        
        # Charts Row 1
        c_col1, c_col2 = st.columns(2)
        
        with c_col1:
            st.markdown("##### Tickets by Priority")
            priority_counts = df_tickets["priority"].value_counts().reset_index()
            priority_counts.columns = ["Priority", "Count"]
            fig_p = px.pie(
                priority_counts, 
                values="Count", 
                names="Priority", 
                color="Priority",
                color_discrete_map={"High": "#ef4444", "Medium": "#f59e0b", "Low": "#10b981"},
                hole=0.4
            )
            fig_p.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#f8fafc")
            st.plotly_chart(fig_p, use_container_width=True)
            
        with c_col2:
            st.markdown("##### Tickets by Category")
            category_counts = df_tickets["category"].value_counts().reset_index()
            category_counts.columns = ["Category", "Count"]
            fig_c = px.bar(
                category_counts, 
                x="Category", 
                y="Count", 
                color="Category",
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig_c.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#f8fafc")
            st.plotly_chart(fig_c, use_container_width=True)

        st.write("")
        
        # Charts Row 2
        c_col3, c_col4 = st.columns(2)
        with c_col3:
            st.markdown("##### Ticket Inflow Trend")
            # Parse created_at dates to daily trend
            df_tickets["date"] = df_tickets["created_at"].apply(lambda x: x.split(" ")[0])
            trend = df_tickets.groupby("date").size().reset_index(name="Inflow")
            trend = trend.sort_values("date")
            fig_t = px.line(
                trend,
                x="date",
                y="Inflow",
                markers=True,
                color_discrete_sequence=["#a78bfa"]
            )
            fig_t.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#f8fafc")
            st.plotly_chart(fig_t, use_container_width=True)
            
        with c_col4:
            st.markdown("##### Status Distribution by Priority")
            fig_sp = px.histogram(
                df_tickets,
                x="priority",
                color="status",
                barmode="group",
                color_discrete_map={"Open": "#3b82f6", "In Progress": "#f59e0b", "Resolved": "#10b981"},
                category_orders={"priority": ["Low", "Medium", "High"]}
            )
            fig_sp.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#f8fafc")
            st.plotly_chart(fig_sp, use_container_width=True)

# TAB 3: Database Console
with tab3:
    st.subheader("SQLite Ticket Console")
    
    # Filter Controls
    f_col1, f_col2, f_col3 = st.columns([1, 1, 2])
    with f_col1:
        sel_status = st.selectbox("Filter Status", ["All", "Open", "In Progress", "Resolved"])
    with f_col2:
        sel_priority = st.selectbox("Filter Priority", ["All", "Low", "Medium", "High"])
    
    status_filter = None if sel_status == "All" else sel_status
    priority_filter = None if sel_priority == "All" else sel_priority
    
    tickets_list = list_all_tickets(status=status_filter, priority=priority_filter)
    
    if not tickets_list:
        st.info("No tickets match the selected filters.")
    else:
        df_list = pd.DataFrame(tickets_list)
        st.dataframe(df_list, use_container_width=True)
        
    st.markdown("---")
    st.subheader("Manual Actions")
    
    act_col1, act_col2 = st.columns(2)
    
    with act_col1:
        with st.form("manual_create_ticket_form"):
            st.markdown("##### Create Support Ticket")
            c_name = st.text_input("Customer Name")
            c_email = st.text_input("Customer Email")
            c_subject = st.text_input("Subject")
            c_desc = st.text_area("Description")
            c_pri = st.selectbox("Priority", ["Low", "Medium", "High"])
            c_cat = st.text_input("Category", value="General")
            
            submitted = st.form_submit_button("Submit Ticket")
            if submitted:
                if c_name and c_email and c_subject and c_desc:
                    tid = create_ticket(c_name, c_email, c_subject, c_desc, priority=c_pri, category=c_cat)
                    st.success(f"Successfully created ticket {tid}!")
                    st.rerun()
                else:
                    st.error("All fields must be filled to create a ticket.")
                    
    with act_col2:
        with st.form("manual_update_ticket_form"):
            st.markdown("##### Update Ticket Status / Resolution")
            u_id = st.text_input("Ticket ID (e.g. TKT-1001)")
            u_status = st.selectbox("New Status", ["Open", "In Progress", "Resolved"])
            u_notes = st.text_area("Resolution / Status Update Notes")
            
            submitted_update = st.form_submit_button("Update Ticket")
            if submitted_update:
                if u_id:
                    success = update_ticket(u_id, status=u_status, resolution_notes=u_notes)
                    if success:
                        st.success(f"Ticket {u_id} updated successfully!")
                        st.rerun()
                    else:
                        st.error(f"Failed to update ticket {u_id}. Verify ID.")
                else:
                    st.error("Ticket ID is required.")

# TAB 4: RAG Knowledge Base
with tab4:
    st.subheader("FAISS RAG Knowledge Base Console")
    
    if not (os.path.exists(FAISS_INDEX_PATH) and os.path.exists(METADATA_PATH)):
        st.error("FAISS index or metadata pickle not found. Run db_manager.py first.")
    else:
        st.success("FAISS index successfully loaded offline. Model used: sentence-transformers/all-MiniLM-L6-v2")
        
        # Test Query
        kb_query = st.text_input("Test Query Vector Search:", placeholder="e.g. How to cancel ticket for game")
        if kb_query:
            with st.spinner("Searching FAISS vector space..."):
                searcher = VectorSearcher()
                results = searcher.search(kb_query, top_k=3)
                
                if not results:
                    st.info("No matching database vectors found.")
                else:
                    st.markdown("##### Top matches from Vector Search:")
                    for i, res in enumerate(results):
                        st.markdown(f"""
                        **Match #{i+1}** (Distance Similarity: `{res['distance']:.4f}`)
                        - **Intent**: `{res['intent']}`
                        - **Category**: `{res['category']}`
                        - **Sample Question**: *"{res['instruction']}"*
                        - **Template Response**: 
                        ```
                        {res['response']}
                        ```
                        """)
