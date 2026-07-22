import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import time

from rag_engine import init_vector_db, query_rag
from agent_engine import stream_nemotron_agent, web_search, PRIMARY_MODEL, FALLBACK_MODELS

# Page configuration with modern wide layout and dark theme
st.set_page_config(
    page_title="Insurance AI Agent Ultra - NVIDIA Nemotron RAG",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (CSS Injection for WOW Modern Design)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #090d16 0%, #101726 50%, #0c111e 100%);
        color: #f3f4f6;
    }
    
    /* Glowing Title Banner */
    .hero-container {
        background: linear-gradient(90deg, rgba(16, 185, 129, 0.18) 0%, rgba(99, 102, 241, 0.18) 50%, rgba(236, 72, 153, 0.18) 100%);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 20px;
        padding: 28px 36px;
        margin-bottom: 24px;
        box-shadow: 0 10px 40px 0 rgba(0, 0, 0, 0.45);
        backdrop-filter: blur(12px);
    }
    
    .hero-title {
        font-family: 'Outfit', sans-serif;
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #10b981 0%, #6366f1 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
        letter-spacing: -0.5px;
    }
    
    .hero-subtitle {
        color: #9ca3af;
        font-size: 1.1rem;
    }
    
    .badge-pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 8px;
        background: rgba(99, 102, 241, 0.2);
        border: 1px solid rgba(99, 102, 241, 0.4);
        color: #a5b4fc;
    }

    /* Reasoning Box Styling */
    .reasoning-box {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.9) 100%);
        border-left: 4px solid #6366f1;
        padding: 16px 20px;
        border-radius: 10px;
        margin-bottom: 14px;
        font-family: 'Courier New', Courier, monospace;
        font-size: 0.9rem;
        color: #c7d2fe;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.3);
    }
    
    /* RAG Card */
    .rag-card {
        background: rgba(17, 24, 39, 0.75);
        border: 1px solid rgba(16, 185, 129, 0.35);
        border-radius: 12px;
        padding: 14px 18px;
        margin-top: 10px;
        margin-bottom: 10px;
        transition: all 0.3s ease;
    }
    .rag-card:hover {
        border-color: rgba(16, 185, 129, 0.8);
        box-shadow: 0 4px 20px rgba(16, 185, 129, 0.15);
    }
    
    .rag-score {
        background: linear-gradient(90deg, #10b981 0%, #059669 100%);
        color: #ffffff;
        font-weight: 700;
        padding: 3px 10px;
        border-radius: 14px;
        font-size: 0.78rem;
    }
    
    /* Policy Card */
    .policy-card {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 16px;
    }
    .policy-card:hover {
        border-color: rgba(99, 102, 241, 0.5);
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #4f46e5 0%, #7c3aed 100%);
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: 600;
        padding: 8px 16px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.45);
    }
</style>
""", unsafe_allow_html=True)

# Dataset Path
CSV_PATH = r"d:\AMINE2\COURS FAC\LEARNING\YOUTUBER\Codes\10-InsuranceChatbot\insurance.csv"

@st.cache_resource
def get_db():
    return init_vector_db(force_rebuild=False)

@st.cache_data
def get_df_sample():
    return pd.read_csv(CSV_PATH)

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "👋 Welcome to **InsuranceBot Ultra**! I am your advanced AI Concierge powered by **NVIDIA Nemotron-3-Super-120B**, ChromaDB Vector Search, and live market intelligence. How can I assist you with your insurance inquiries today?"}
    ]

# Header Banner
st.markdown("""
<div class="hero-container">
    <div class="hero-title">🛡️ Insurance AI Agent & RAG Ultra Dashboard</div>
    <div class="hero-subtitle">
        <span class="badge-pill">⚡ NVIDIA Nemotron-3-Super-120B</span>
        <span class="badge-pill">🧠 Thinking Reasoning Enabled</span>
        <span class="badge-pill">📚 RAG Vector Database</span>
        <span class="badge-pill">🌐 Live Web Search</span>
        <span class="badge-pill">📝 Claim Simulator</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar Control Panel
with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/insurance.png", width=70)
    st.title("⚙️ Agent Command Center")
    
    st.markdown("---")
    st.subheader("🤖 Model & Fallback Selection")
    
    model_options = [PRIMARY_MODEL] + FALLBACK_MODELS
    selected_model = st.selectbox("Active NVIDIA Model", options=model_options, index=0)
    
    temperature = st.slider("Temperature (Creativity)", 0.0, 1.5, 0.7, 0.1)
    top_p = st.slider("Top-P", 0.1, 1.0, 0.95, 0.05)
    reasoning_budget = st.slider("Reasoning Budget (Tokens)", 1024, 32768, 16384, 1024)
    
    st.markdown("---")
    st.subheader("🔍 RAG Knowledge Retrieval")
    enable_rag = st.checkbox("Enable RAG Vector Search", value=True)
    rag_k = st.slider("RAG Context Docs (k)", 1, 8, 4)
    category_filter = st.selectbox("Category Filter", ["ALL", "AUTO_INSURANCE", "HEALTH_INSURANCE", "LIFE_INSURANCE", "HOME_INSURANCE", "CLAIMS", "PAYMENT", "COVERAGE"])
    
    st.markdown("---")
    st.subheader("🌐 Real-time Web Search")
    enable_web_search = st.checkbox("Enable Live DuckDuckGo Web Search", value=False)
    
    st.markdown("---")
    st.subheader("📥 Export & Reset")
    
    # Export Chat Session
    chat_export_json = json.dumps(st.session_state.messages, indent=2)
    st.download_button(
        label="📥 Export Chat History (JSON)",
        data=chat_export_json,
        file_name="insurance_chat_session.json",
        mime="application/json"
    )
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = [
            {"role": "assistant", "content": "Conversation history cleared. How may I help you today?"}
        ]
        st.rerun()

# Navigation Tabs
tab_chat, tab_analytics, tab_calculator, tab_claim, tab_comparison, tab_vectordb = st.tabs([
    "💬 AI Chatbot & Agent",
    "📊 Dataset Analytics",
    "🧮 Premium Quote Estimator",
    "📝 AI Claim Assistant",
    "⚖️ Policy Comparison Tool",
    "🔍 Vector DB Inspector"
])

# ==========================================
# TAB 1: AI CHATBOT & REASONING AGENT
# ==========================================
with tab_chat:
    st.subheader("💬 Interactive AI Agent Concierge")
    
    # Smart Prompt Suggestions
    st.markdown("**💡 Click a Smart Suggestion to test:**")
    col1, col2, col3, col4 = st.columns(4)
    prompt_preset = None
    with col1:
        if st.button("🚘 View Auto Insurance"):
            prompt_preset = "How can I view my auto insurance policy details on the portal?"
    with col2:
        if st.button("📋 File a Health Claim"):
            prompt_preset = "What is the exact step by step process to submit a health insurance claim?"
    with col3:
        if st.button("💳 Premium Payment Methods"):
            prompt_preset = "What options do I have for paying my monthly insurance premium?"
    with col4:
        if st.button("🌐 2026 Insurance Trends"):
            prompt_preset = "What are the latest AI insurance trends and regulations in 2026?"
            
    st.markdown("---")
    
    # Render Chat Messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if "model_used" in msg and msg["model_used"]:
                st.caption(f"🤖 **Model Used:** `{msg['model_used']}`")
            if "reasoning" in msg and msg["reasoning"]:
                with st.expander("🧠 View Nemotron Agent Reasoning & Thinking Process"):
                    st.markdown(f"<div class='reasoning-box'>{msg['reasoning']}</div>", unsafe_allow_html=True)
            st.markdown(msg["content"])
            if "rag_docs" in msg and msg["rag_docs"]:
                with st.expander(f"📚 Retreived RAG Context ({len(msg['rag_docs'])} documents matched)"):
                    for doc in msg["rag_docs"]:
                        st.markdown(f"""
                        <div class="rag-card">
                            <span class="rag-score">Similarity: {doc['score']}</span> <b>Category:</b> {doc['category']} | <b>Intent:</b> {doc['intent']}
                            <br><b>Matched User Query:</b> {doc['instruction']}
                            <br><b>Official Answer:</b> {doc['response'][:250]}...
                        </div>
                        """, unsafe_allow_html=True)

    # Input Box
    user_input = st.chat_input("Ask any insurance question (auto, health, life, claims, policies)...")
    if prompt_preset:
        user_input = prompt_preset
        
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
            
        # RAG Retrieval
        rag_docs = []
        rag_text = ""
        if enable_rag:
            try:
                coll = get_db()
                rag_docs = query_rag(coll, user_input, n_results=rag_k, category_filter=category_filter)
                if rag_docs:
                    rag_text = "\n\n".join([f"Source {i+1} [{d['category']}/{d['intent']}]: Q: {d['instruction']} | A: {d['response']}" for i, d in enumerate(rag_docs)])
            except Exception as e:
                st.warning(f"RAG search error: {str(e)}")
                
        # Web Search
        web_text = ""
        if enable_web_search:
            with st.spinner("Searching live web sources..."):
                web_text = web_search(user_input, max_results=3)
                
        # Stream response
        with st.chat_message("assistant"):
            reasoning_placeholder = st.empty()
            content_placeholder = st.empty()
            
            full_reasoning = ""
            full_content = ""
            active_model_used = selected_model
            
            history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[:-1]]
            history.append({"role": "user", "content": user_input})
            
            with st.spinner(f"Agent ({selected_model}) is thinking and generating response..."):
                try:
                    stream = stream_nemotron_agent(
                        messages=history,
                        rag_context=rag_text,
                        web_context=web_text,
                        temperature=temperature,
                        top_p=top_p,
                        reasoning_budget=reasoning_budget,
                        selected_model=selected_model
                    )
                    
                    for chunk in stream:
                        if chunk.get("active_model"):
                            active_model_used = chunk["active_model"]
                        if chunk.get("reasoning"):
                            full_reasoning += chunk["reasoning"]
                            with reasoning_placeholder.expander(f"🧠 View Reasoning Process ({active_model_used})", expanded=True):
                                st.markdown(f"<div class='reasoning-box'>{full_reasoning}</div>", unsafe_allow_html=True)
                        if chunk.get("content"):
                            full_content += chunk["content"]
                            content_placeholder.markdown(full_content + "▌")
                            
                    content_placeholder.markdown(full_content)
                    st.caption(f"🤖 **Model Used:** `{active_model_used}`")
                except Exception as e:
                    st.error(f"Error calling model API: {str(e)}")
                    full_content = f"I encountered an API error: {str(e)}"
                
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_content,
                "reasoning": full_reasoning,
                "rag_docs": rag_docs,
                "model_used": active_model_used
            })

# ==========================================
# TAB 2: DATASET & INTENT ANALYTICS
# ==========================================
with tab_analytics:
    st.subheader("📊 Dataset Insights & Intent Distribution")
    df = get_df_sample()
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Indexed Knowledge Records", f"{len(df):,}")
    m2.metric("Unique Customer Intents", f"{df['intent'].nunique()}")
    m3.metric("Insurance Categories", f"{df['category'].nunique()}")
    m4.metric("Average Response Length", f"{int(df['response'].str.len().mean())} chars")
    
    st.markdown("---")
    
    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("### Category Distribution")
        cat_counts = df['category'].value_counts().reset_index()
        cat_counts.columns = ['Category', 'Count']
        fig_cat = px.bar(
            cat_counts, x='Category', y='Count', color='Count',
            color_continuous_scale='Viridis', title="Records per Category"
        )
        fig_cat.update_layout(template="plotly_dark", height=380)
        st.plotly_chart(fig_cat, use_container_width=True)
        
    with col_right:
        st.markdown("### Top Customer Intents")
        intent_counts = df['intent'].value_counts().head(10).reset_index()
        intent_counts.columns = ['Intent', 'Count']
        fig_intent = px.pie(
            intent_counts, names='Intent', values='Count', hole=0.45,
            title="Top 10 Intent Share", color_discrete_sequence=px.colors.qualitative.Bold
        )
        fig_intent.update_layout(template="plotly_dark", height=380)
        st.plotly_chart(fig_intent, use_container_width=True)
        
    st.markdown("---")
    st.markdown("### Dataset Raw Explorer")
    st.dataframe(df[['category', 'intent', 'instruction', 'response']].head(100), use_container_width=True)

# ==========================================
# TAB 3: PREMIUM & COVERAGE CALCULATOR
# ==========================================
with tab_calculator:
    st.subheader("🧮 Interactive Premium & Underwriting Quote Calculator")
    
    calc_c1, calc_c2 = st.columns([1, 1])
    with calc_c1:
        st.markdown("#### Customer Risk Factors")
        policy_type = st.selectbox("Policy Type", ["Auto Insurance", "Health Insurance", "Life Insurance", "Home Insurance"])
        age = st.slider("Policyholder Age", 18, 85, 32)
        coverage_amount = st.select_slider("Coverage Limit ($)", options=[50000, 100000, 250000, 500000, 1000000], value=250000)
        deductible = st.select_slider("Deductible ($)", options=[250, 500, 1000, 2500, 5000], value=500)
        claims_history = st.radio("Prior Claims (Past 3 Years)", ["0 Claims (Clean)", "1 Claim", "2+ Claims"])
        
        # Calculation Logic
        base_rate = 60.0
        if policy_type == "Auto Insurance":
            base_rate = 85.0
            if age < 25 or age > 70: base_rate *= 1.45
        elif policy_type == "Health Insurance":
            base_rate = 120.0 + (age * 3.8)
        elif policy_type == "Life Insurance":
            base_rate = 35.0 + (age * 4.5)
        elif policy_type == "Home Insurance":
            base_rate = 70.0
            
        coverage_factor = (coverage_amount / 100000.0) * 18.0
        deductible_discount = (500.0 - deductible) * 0.02
        claims_penalty = 0.0 if "0" in claims_history else (40.0 if "1" in claims_history else 90.0)
        
        monthly_premium = max(30.0, round(base_rate + coverage_factor - deductible_discount + claims_penalty, 2))
        annual_premium = round(monthly_premium * 12, 2)
        risk_score = min(100, int((age/85)*25 + (claims_penalty/90)*55 + (coverage_amount/1000000)*20))
        
    with calc_c2:
        st.markdown("#### Estimated Quote Results")
        res_c1, res_c2 = st.columns(2)
        res_c1.metric("Estimated Monthly Premium", f"${monthly_premium:.2f}")
        res_c2.metric("Estimated Annual Premium", f"${annual_premium:.2f}")
        
        fig_risk = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = risk_score,
            title = {'text': "Risk Tier Rating (0-100)"},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': "#6366f1"},
                'steps': [
                    {'range': [0, 35], 'color': "rgba(16, 185, 129, 0.35)"},
                    {'range': [35, 70], 'color': "rgba(245, 158, 11, 0.35)"},
                    {'range': [70, 100], 'color': "rgba(239, 68, 68, 0.35)"}
                ]
            }
        ))
        fig_risk.update_layout(template="plotly_dark", height=280)
        st.plotly_chart(fig_risk, use_container_width=True)
        
        if st.button("💬 Discuss Quote with AI Agent"):
            quote_query = f"Can you analyze my estimated {policy_type} quote of ${monthly_premium:.2f}/month (Age: {age}, Coverage: ${coverage_amount}) and advise how I can lower my premium?"
            st.session_state.messages.append({"role": "user", "content": quote_query})
            st.rerun()

# ==========================================
# TAB 4: AI CLAIM ASSISTANT & SIMULATOR
# ==========================================
with tab_claim:
    st.subheader("📝 Intelligent First Notice of Loss (FNOL) Claim Filing Assistant")
    st.markdown("Simulate submitting an insurance claim and receive immediate AI risk validation and document checklists.")
    
    with st.form("claim_form"):
        form_c1, form_c2 = st.columns(2)
        with form_c1:
            claim_type = st.selectbox("Incident Category", ["Auto Accident", "Medical / Hospitalization", "Property Damage", "Theft", "Natural Disaster"])
            incident_date = st.date_input("Incident Date")
            estimated_loss = st.number_input("Estimated Financial Loss ($)", min_value=100, max_value=500000, value=2500)
        with form_c2:
            police_report = st.radio("Is Police / Official Report Filed?", ["Yes", "No", "In Progress"])
            incident_desc = st.text_area("Detailed Description of Incident", placeholder="Describe what happened, location, parties involved, and damage incurred...")
            
        submit_claim = st.form_submit_button("🚀 Analyze & Generate Claim Packet")
        
    if submit_claim:
        st.success("Claim Packet Successfully Analyzed!")
        st.markdown("### 📋 AI Claim Triage Summary")
        
        triage_c1, triage_c2, triage_c3 = st.columns(3)
        triage_c1.metric("Recommended Priority", "HIGH" if estimated_loss > 10000 or police_report == "No" else "MEDIUM")
        triage_c2.metric("Fast-Track Eligible", "YES" if estimated_loss < 3000 and police_report == "Yes" else "NO (Requires Manual Adjuster)")
        triage_c3.metric("Required Verification Docs", "3 Documents")
        
        st.markdown("#### Required Documentation Checklist:")
        st.markdown("""
        - [ ] Photo / Video evidence of incident scene & damages
        - [ ] Official Police/Medical Incident Report (ID Ref)
        - [ ] Repair Estimates / Itemized Medical Invoices
        """)
        
        if st.button("💬 Send Claim Summary to AI Agent for Formal Review"):
            claim_prompt = f"I am filing a {claim_type} claim for estimated loss ${estimated_loss}. Official report: {police_report}. Description: '{incident_desc}'. What are the exact steps and guidelines I should follow now?"
            st.session_state.messages.append({"role": "user", "content": claim_prompt})
            st.rerun()

# ==========================================
# TAB 5: POLICY COMPARISON TOOL
# ==========================================
with tab_comparison:
    st.subheader("⚖️ Side-by-Side Policy Comparison Matrix")
    st.markdown("Compare feature coverage across standard tier plans.")
    
    comp_data = {
        "Feature / Coverage": [
            "Monthly Base Premium",
            "Medical Expenses Limit",
            "Property Damage Liability",
            "Collision Deductible",
            "Roadside Assistance (24/7)",
            "Rental Car Coverage",
            "Zero Deductible Glass Repair"
        ],
        "Basic Shield Tier": ["$45 / mo", "$25,000", "$15,000", "$1,000", "❌ Excluded", "❌ Excluded", "❌ Excluded"],
        "Gold Standard Tier": ["$95 / mo", "$100,000", "$50,000", "$500", "✅ Included", "✅ $30/day", "✅ Included"],
        "Platinum Ultra Tier": ["$160 / mo", "$250,000", "$100,000", "$250", "✅ VIP Priority", "✅ $60/day", "✅ Included"]
    }
    
    df_comp = pd.DataFrame(comp_data)
    st.table(df_comp)
    
    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        if st.button("Inquire about Basic Tier"):
            st.session_state.messages.append({"role": "user", "content": "Tell me more about the Basic Shield Insurance Tier and who it is best suited for."})
            st.rerun()
    with col_p2:
        if st.button("Inquire about Gold Tier"):
            st.session_state.messages.append({"role": "user", "content": "What are the main advantages of upgrading to the Gold Standard Insurance Tier?"})
            st.rerun()
    with col_p3:
        if st.button("Inquire about Platinum Tier"):
            st.session_state.messages.append({"role": "user", "content": "What premium benefits are included in the Platinum Ultra Tier?"})
            st.rerun()

# ==========================================
# TAB 6: VECTOR DB INSPECTOR
# ==========================================
with tab_vectordb:
    st.subheader("🔍 Vector DB Semantic Search & Document Explorer")
    st.markdown("Perform similarity queries against ChromaDB to inspect vector index embeddings and cosine similarity scores.")
    
    search_col1, search_col2 = st.columns([3, 1])
    with search_col1:
        db_query = st.text_input("Semantic Search Query", value="How do I view my auto insurance policy document?")
    with search_col2:
        top_k_val = st.slider("Top Results", 1, 10, 5)
        
    if st.button("🔎 Execute Vector Database Query"):
        try:
            coll = get_db()
            results = query_rag(coll, db_query, n_results=top_k_val)
            st.markdown(f"Found **{len(results)}** vector matches:")
            for i, r in enumerate(results):
                st.markdown(f"""
                <div class="rag-card">
                    <h4>#{i+1} | Score: <span class="rag-score">{r['score']}</span> | Category: {r['category']}</h4>
                    <p><b>Intent:</b> <code>{r['intent']}</code></p>
                    <p><b>Instruction:</b> {r['instruction']}</p>
                    <p><b>Official Procedure Response:</b> {r['response']}</p>
                </div>
                """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"ChromaDB lookup failed: {str(e)}")
