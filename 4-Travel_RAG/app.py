import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

import streamlit as st
import pandas as pd
import numpy as np
import time
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Configure the Streamlit Page
st.set_page_config(
    page_title="AeroQuest | Semantic RAG Chatbot",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium CSS Injection
st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

    /* Global styles */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif;
        font-weight: 600;
        letter-spacing: -0.5px;
    }

    /* Main container styling */
    .stApp {
        background: linear-gradient(135deg, #0e1117 0%, #0a0b10 100%);
        color: #e2e8f0;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: rgba(17, 22, 34, 0.95);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
    }
    [data-testid="stSidebar"] .stMarkdown {
        color: #94a3b8;
    }

    /* Custom Title Area */
    .hero-title {
        background: linear-gradient(90deg, #38bdf8 0%, #a855f7 50%, #f43f5e 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 3rem;
        margin-bottom: 0.2rem;
    }
    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 2rem;
        font-weight: 300;
    }

    /* Glassmorphism Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 20px;
        backdrop-filter: blur(12px);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        margin-bottom: 1.5rem;
        transition: all 0.3s ease;
    }
    .glass-card:hover {
        border-color: rgba(56, 189, 248, 0.3);
        box-shadow: 0 8px 32px 0 rgba(56, 189, 248, 0.05);
    }

    /* Chat bubble container */
    .chat-container {
        display: flex;
        flex-direction: column;
        gap: 16px;
        margin-bottom: 2rem;
    }

    /* Chat bubbles */
    .chat-bubble {
        display: flex;
        padding: 16px 20px;
        border-radius: 16px;
        max-width: 85%;
        line-height: 1.6;
        font-size: 0.95rem;
        position: relative;
        animation: fadeIn 0.3s ease;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .chat-bubble.user {
        background: linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        align-self: flex-end;
        color: #ffffff;
        border-bottom-right-radius: 4px;
        box-shadow: 0 4px 15px rgba(29, 78, 216, 0.2);
    }

    .chat-bubble.assistant {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.05);
        align-self: flex-start;
        color: #e2e8f0;
        border-bottom-left-radius: 4px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }

    .bubble-avatar {
        font-size: 1.3rem;
        margin-right: 12px;
        display: flex;
        align-items: center;
    }

    .bubble-content {
        flex: 1;
    }

    /* Stat values */
    .stat-val {
        font-size: 2rem;
        font-weight: 700;
        color: #38bdf8;
        font-family: 'Outfit', sans-serif;
    }
    .stat-label {
        font-size: 0.8rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Scrollbars */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: rgba(0, 0, 0, 0.1);
    }
    ::-webkit-scrollbar-thumb {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(255, 255, 255, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# ----------------- VECTOR STORE LOADER -----------------

INDEX_PATH = r"d:\AMINE2\COURS FAC\LEARNING\YOUTUBER\Codes\AI_Agent_RAG\faiss_index"

@st.cache_resource(show_spinner="Loading Semantic Vector Store (all-MiniLM-L6-v2)...")
def load_vector_store():
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    vectorstore = FAISS.load_local(
        INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vectorstore

try:
    vectorstore = load_vector_store()
    # Simple cached stats for display
    total_docs = vectorstore.index.ntotal
except Exception as e:
    st.error(f"Error loading FAISS index: {e}")
    st.stop()

# ----------------- RAG ENGINE -----------------

def retrieve_rag_context(query, k=3):
    # Perform similarity search with score
    results = vectorstore.similarity_search_with_score(query, k=k)
    
    formatted_results = []
    for doc, distance in results:
        # L2 distance to similarity score conversion
        similarity = 1.0 / (1.0 + distance)
        formatted_results.append({
            'instruction': doc.page_content,
            'intent': doc.metadata.get('intent', 'N/A'),
            'category': doc.metadata.get('category', 'N/A'),
            'response': doc.metadata.get('response', 'N/A'),
            'similarity': similarity
        })
    return formatted_results

# ----------------- SIDEBAR -----------------

with st.sidebar:
    st.markdown("<div style='text-align: center; margin-top: 1rem;'><span style='font-size: 3rem;'>✈️</span></div>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; color: white;'>AeroQuest Control Panel</h2>", unsafe_allow_html=True)
    st.markdown("<hr style='border-color: rgba(255,255,255,0.05);'>", unsafe_allow_html=True)
    
    # Knowledge Base Stats
    st.markdown("### 📊 Semantic RAG Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"<div class='stat-val'>{total_docs:,}</div><div class='stat-label'>Embeddings</div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='stat-val'>MiniLM</div><div class='stat-label'>Model</div>", unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Info Panel
    st.markdown("### 🔍 RAG Architecture")
    st.markdown("""
    - **Embeddings**: `all-MiniLM-L6-v2` (384-dimensional dense vectors)
    - **Vector DB**: `FAISS` (Facebook AI Similarity Search)
    - **Retrieval**: Semantic Similarity (cosine/L2 distance matching)
    - **LLM**: Groq Llama-3.3-70B-Versatile
    """)
    
    st.markdown("<hr style='border-color: rgba(255,255,255,0.05);'>", unsafe_allow_html=True)
    
    # Reset Chat button
    if st.button("🔄 Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ----------------- MAIN INTERFACE -----------------

st.markdown("<div class='hero-title'>AeroQuest</div>", unsafe_allow_html=True)
st.markdown("<div class='hero-subtitle'>Semantic RAG Assistant powered by FAISS & Llama-3.3-70b-versatile</div>", unsafe_allow_html=True)

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History using Custom Bubbles
chat_container_html = "<div class='chat-container'>"
for msg in st.session_state.messages:
    role = msg["role"]
    avatar = "👤" if role == "user" else "🤖"
    chat_container_html += f"<div class='chat-bubble {role}'><div class='bubble-avatar'>{avatar}</div><div class='bubble-content'>{msg['content']}</div></div>"
chat_container_html += "</div>"
st.markdown(chat_container_html, unsafe_allow_html=True)

# Display suggestions if chat is empty
if not st.session_state.messages:
    st.markdown("<div style='margin-bottom: 2rem;'></div>", unsafe_allow_html=True)
    st.markdown("### 💡 Quick Suggestions")
    cols = st.columns(4)
    suggestions = [
        ("🧳 Baggage Allowance", "How do I check my baggage allowance?"),
        ("🎫 Boarding Pass", "How can I print my boarding pass?"),
        ("🔄 Change Booking", "Can I change my flight booking?"),
        ("❌ Cancel Trip", "How do I cancel my trip?")
    ]
    for idx, (label, query) in enumerate(suggestions):
        with cols[idx]:
            if st.button(label, key=f"sugg_{idx}", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": query})
                st.rerun()

# ----------------- PROMPT INPUT & LLM QUERY -----------------

user_input = st.chat_input("Ask about flight status, baggage, cancellation, check-in, etc...")

if user_input:
    # Append user question
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.rerun()

# Check if last message is from user to trigger AI response
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    last_query = st.session_state.messages[-1]["content"]
    
    with st.spinner("Retrieving semantic matches & thinking..."):
        # 1. Retrieve context via FAISS semantic search
        retrieved_records = retrieve_rag_context(last_query, k=3)
        
        # Format RAG Context for Prompt
        context_str = ""
        for i, rec in enumerate(retrieved_records, 1):
            context_str += f"Record #{i} (Intent: {rec['intent']}, Category: {rec['category']}):\n"
            context_str += f"User inquiry pattern: {rec['instruction']}\n"
            context_str += f"Response Template: {rec['response']}\n\n"
        
        # 2. Build system instructions
        system_instruction = """You are AeroQuest's Virtual Customer Assistant, a highly professional, polite, and helpful travel support agent.
Your goal is to answer the customer's query using the provided retrieved templates from our Knowledge Base.

Important Instructions:
- Adopt a warm, professional, and customer-first tone.
- When generating the response, replace placeholders in the templates (e.g. {{WEBSITE_URL}}, {{APP_NAME}}, {{BOOKINGS_OPTION}}, {{CHECKIN_SECTION}}, {{CUSTOMER_SUPPORT}}) with realistic brand-specific values:
  - {{WEBSITE_URL}} -> aeroquest-travel.com
  - {{APP_NAME}} -> AeroQuest Mobile App
  - {{BOOKINGS_OPTION}} -> "My Trips" dashboard
  - {{CHECKIN_SECTION}} -> "Online Check-In" portal
  - {{CUSTOMER_SUPPORT}} -> AeroQuest Customer Care Support Team
  - {{FLIGHT_NUMBER}} -> flight details
- Provide a direct, step-by-step resolution based on the matching templates. Do not state "Based on the retrieved templates..." to the user. Make it sound natural and authoritative.
- If the query is completely unrelated to our travel support categories (e.g., coding, general science), politely inform the customer that you are specialized in travel support, booking, baggage, and check-in help, and guide them back to travel queries.
"""

        # 3. Call Groq via LangChain LCEL Chain
        try:
            llm = ChatGroq(
                groq_api_key="",
                model_name="llama-3.3-70b-versatile",
                temperature=0.3,
                max_tokens=1024
            )
            
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", system_instruction),
                ("user", "Customer Query: \"{query}\"\n\nRetrieved Knowledge Base Context:\n\"\"\"\n{context}\n\"\"\"\n\nPlease write the final assistant response answering the Customer Query, incorporating the retrieved templates where applicable:")
            ])
            
            chain = prompt_template | llm | StrOutputParser()
            assistant_response = chain.invoke({
                "query": last_query,
                "context": context_str if context_str else "No highly similar template records found."
            })
            
        except Exception as err:
            assistant_response = f"⚠️ API Error: Unable to reach Groq services. Details: {err}"
        
        # Append assistant response
        st.session_state.messages.append({"role": "assistant", "content": assistant_response, "retrieved": retrieved_records})
        st.rerun()

# ----------------- RAG INSPECTOR WIDGET -----------------

# Display RAG inspection panel if the last message has retrieved context
if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
    last_msg = st.session_state.messages[-1]
    if "retrieved" in last_msg and last_msg["retrieved"]:
        st.markdown("### 🔍 Semantic RAG Retrieval Inspector")
        st.markdown("These are the most semantically similar records retrieved from the `FAISS` vector database:")
        
        cols = st.columns(len(last_msg["retrieved"]))
        for idx, rec in enumerate(last_msg["retrieved"]):
            with cols[idx]:
                card_html = (
                    f"<div class='glass-card' style='height: 100%;'>"
                    f"<div style='display: flex; justify-content: space-between; align-items: center;'>"
                    f"<span style='background: #1e293b; color: #38bdf8; padding: 4px 8px; border-radius: 8px; font-size: 0.75rem; font-weight: bold;'>Rank #{idx+1}</span>"
                    f"<span style='color: #4ade80; font-size: 0.8rem; font-weight: bold;'>Sim: {rec['similarity']:.3f}</span>"
                    f"</div>"
                    f"<h4 style='margin: 10px 0 5px 0; font-size: 1rem; color: #f1f5f9;'>{rec['intent']}</h4>"
                    f"<p style='color: #94a3b8; font-size: 0.75rem; text-transform: uppercase;'>Category: {rec['category']}</p>"
                    f"<hr style='border-color: rgba(255,255,255,0.05); margin: 8px 0;'>"
                    f"<p style='font-size: 0.8rem; color: #cbd5e1; font-style: italic;'>\"{rec['instruction']}\"</p>"
                    f"</div>"
                )
                st.markdown(card_html, unsafe_allow_html=True)
