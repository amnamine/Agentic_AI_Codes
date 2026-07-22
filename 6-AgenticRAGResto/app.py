import streamlit as st
import pandas as pd
import os
import time
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# Page Config
st.set_page_config(
    page_title="GourmetAgent AI - Premium Restaurant Concierge",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Glassmorphism & Neon Glow Aesthetics)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"], .stApp {
        font-family: 'Outfit', sans-serif;
        background-color: #0c0f1d;
        color: #e2e8f0;
    }
    
    /* Neon Glow Gradient Header */
    .header-container {
        background: linear-gradient(135deg, rgba(26,21,44,0.85) 0%, rgba(13,11,28,0.9) 100%);
        padding: 2.5rem;
        border-radius: 20px;
        border: 1px solid rgba(139, 92, 246, 0.3);
        box-shadow: 0 8px 32px 0 rgba(139, 92, 246, 0.15);
        margin-bottom: 2rem;
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    
    .header-container::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(139,92,246,0.1) 0%, transparent 70%);
        z-index: 0;
    }
    
    .title-gradient {
        background: linear-gradient(90deg, #a78bfa, #ec4899, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
    }
    
    .subtitle {
        color: #94a3b8;
        font-size: 1.15rem;
        font-weight: 300;
        letter-spacing: 0.5px;
    }

    /* Glassmorphic Cards & Containers */
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 700;
        color: #f472b6 !important;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #080b16 !important;
        border-right: 1px solid rgba(139, 92, 246, 0.2);
    }
    
    /* Input aesthetics */
    .stTextInput input {
        background-color: rgba(17, 24, 39, 0.8) !important;
        border: 1px solid rgba(139, 92, 246, 0.2) !important;
        color: #f8fafc !important;
        border-radius: 12px !important;
        transition: all 0.3s ease;
    }
    
    .stTextInput input:focus {
        border-color: #ec4899 !important;
        box-shadow: 0 0 10px rgba(236, 72, 153, 0.4) !important;
    }

    /* Custom Chat aesthetics */
    .chat-bubble {
        padding: 1rem 1.25rem;
        border-radius: 16px;
        margin-bottom: 1rem;
        line-height: 1.5;
        border: 1px solid rgba(255, 255, 255, 0.05);
        transition: transform 0.2s ease;
    }
    .chat-bubble:hover {
        transform: translateY(-2px);
    }
    .chat-user {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.15) 0%, rgba(29, 78, 216, 0.2) 100%);
        border-left: 4px solid #3b82f6;
        margin-left: 10%;
    }
    .chat-assistant {
        background: linear-gradient(135deg, rgba(139, 92, 246, 0.15) 0%, rgba(109, 40, 217, 0.2) 100%);
        border-left: 4px solid #8b5cf6;
        margin-right: 10%;
    }
    
    /* Styled lists and badges */
    .badge {
        background: rgba(139, 92, 246, 0.2);
        color: #c084fc;
        padding: 0.2rem 0.6rem;
        border-radius: 50px;
        font-size: 0.75rem;
        font-weight: 600;
        border: 1px solid rgba(139, 92, 246, 0.3);
        margin-right: 0.5rem;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #0c0f1d;
    }
    ::-webkit-scrollbar-thumb {
        background: #3b82f6;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #ec4899;
    }
</style>
""", unsafe_allow_html=True)

# Header Section
st.markdown("""
<div class="header-container">
    <div class="title-gradient">GourmetAgent AI 🍽️</div>
    <div class="subtitle">Next-Gen Restaurant Concierge powered by Agentic RAG & Live Web Search</div>
</div>
""", unsafe_allow_html=True)

# Hardcoded Groq API Key
GROQ_API_KEY = ""
MODEL_NAME = "llama-3.3-70b-versatile"
CSV_PATH = r"d:\AMINE2\COURS FAC\LEARNING\YOUTUBER\Codes\6-AgenticRAGResto\restaurants.csv"
INDEX_PATH = "faiss_index"

# ----------------- VECTOR DATABASE SETUP -----------------

@st.cache_resource(show_spinner=False)
def get_embeddings():
    # Load lightweight but highly accurate sentence transformer model
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource(show_spinner=False)
def load_or_create_vector_db():
    embeddings = get_embeddings()
    
    # If saved index exists, load it
    if os.path.exists(INDEX_PATH):
        try:
            db = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
            return db, True
        except Exception as e:
            st.error(f"Error loading local index: {e}. Recreating...")
            
    # Read restaurants.csv and create index
    if not os.path.exists(CSV_PATH):
        st.error(f"Dataset file restaurants.csv not found at: {CSV_PATH}")
        return None, False
        
    df = pd.read_csv(CSV_PATH)
    
    # Sample the dataset to cover all intents and maintain speed
    # We group by intent and sample 150 rows, or take the whole thing if less than 150
    sampled_df = df.groupby("intent", group_keys=False).apply(
        lambda x: x.sample(min(len(x), 150), random_state=42)
    ).reset_index(drop=True)
    
    texts = sampled_df["instruction"].tolist()
    metadatas = []
    for _, row in sampled_df.iterrows():
        metadatas.append({
            "response": str(row["response"]),
            "intent": str(row["intent"]),
            "category": str(row["category"]),
            "tags": str(row["tags"])
        })
        
    db = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
    # Save index locally
    db.save_local(INDEX_PATH)
    return db, False

with st.spinner("🍽️ Initializing Gourmet Knowledge Base (Agentic RAG)..."):
    db, was_cached = load_or_create_vector_db()

# ----------------- CHATBOT TOOLS SETUP -----------------

# Define tools
@tool
def query_knowledge_base(query: str) -> str:
    """Useful to lookup internal restaurant guidelines, policies, order cancellation procedures, reservation instructions, catering FAQs, and contact info."""
    if not db:
        return "Knowledge base is offline."
    
    docs = db.similarity_search(query, k=3)
    results = []
    for i, doc in enumerate(docs):
        meta = doc.metadata
        res_str = (
            f"Match {i+1} [Intent: {meta.get('intent', 'N/A')}, Category: {meta.get('category', 'N/A')}]:\n"
            f"Q: {doc.page_content}\n"
            f"A: {meta.get('response', 'No response available.')}"
        )
        results.append(res_str)
        
    return "\n\n---\n\n".join(results)

@tool
def search_web_for_restaurants(query: str) -> str:
    """Useful to search the web for external restaurant recommendations, menus, address directions, ratings, and real-time food reviews in specific cities."""
    # Method 1: Library
    try:
        from langchain_community.tools import DuckDuckGoSearchRun
        search = DuckDuckGoSearchRun()
        res = search.run(query)
        if res and "error" not in res.lower() and "failed" not in res.lower():
            return res
    except Exception:
        pass
        
    # Method 2: Fallback direct HTML scraper (highly robust)
    try:
        import requests
        from bs4 import BeautifulSoup
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        url = f"https://html.duckduckgo.com/html/?q={requests.utils.quote(query)}"
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code == 200:
            soup = BeautifulSoup(r.text, "html.parser")
            snippets = soup.find_all("a", class_="result__snippet")
            results = []
            for i, snip in enumerate(snippets[:5]):
                results.append(f"[{i+1}] {snip.get_text().strip()}")
            if results:
                return "\n\n".join(results)
    except Exception as scrape_err:
        return f"Web search failed. Scraper error: {scrape_err}"
        
    return "Web search failed. No results could be retrieved from search engines."

tools = [query_knowledge_base, search_web_for_restaurants]

# Initialize ChatGroq LLM
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name=MODEL_NAME,
    temperature=0.7
)

# Agent setup using LangGraph prebuilt react agent
system_message = (
    "You are GourmetAgent, a premium, highly knowledgeable, and polite AI Restaurant Concierge. "
    "Help the user by answering queries about restaurant catering, reservations, policies, and recommendations. "
    "Use the query_knowledge_base tool to look up internal information about reservation rules, catering cancellations, and restaurant policies. "
    "Use the search_web_for_restaurants tool to get real-time recommendations, reviews, and restaurant info from the web. "
    "Always format your output beautifully with markdown, emojis, lists, or tables where appropriate."
)

agent_executor = create_react_agent(
    model=llm,
    tools=tools,
    prompt=system_message
)

# ----------------- SIDEBAR DESIGN -----------------

with st.sidebar:
    st.markdown("### 🎛️ Agent Control Panel")
    st.markdown("Customize settings and interact with the Gourmet Agent.")
    
    # Active Service Status
    st.markdown("#### ⚡ System Status")
    status_db = "🟢 Online" if db else "🔴 Offline"
    st.markdown(f"**Knowledge Base (RAG):** {status_db}")
    st.markdown(f"**Web Search:** 🟢 Connected")
    st.markdown(f"**Active LLM:** `Llama-3.3-70B`")
    
    st.markdown("---")
    
    # Quick Statistics
    st.markdown("#### 📊 Knowledge Base Stats")
    if os.path.exists(CSV_PATH):
        df_stats = pd.read_csv(CSV_PATH)
        st.metric("Total Ingested Records", f"{len(df_stats):,}")
        st.metric("Unique Customer Intents", f"{df_stats['intent'].nunique()}")
        st.metric("Unique Categories", f"{df_stats['category'].nunique()}")
    
    st.markdown("---")
    
    # Suggested/Quick Prompts
    st.markdown("#### 💡 Quick Suggestion Chips")
    suggestions = [
        "How can I cancel a catering order?",
        "Recommend the best pizza in Rome with great reviews",
        "What is the privacy policy regarding my data?",
        "Find top Japanese sushi places in New York",
        "How do I make a restaurant reservation?",
    ]
    
    # Render quick action buttons
    selected_suggestion = None
    for idx, sug in enumerate(suggestions):
        if st.button(sug, key=f"sug_{idx}", use_container_width=True):
            selected_suggestion = sug

    st.markdown("---")
    if st.button("🧹 Clear Conversation History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ----------------- MAIN CHAT INTERFACE -----------------

# Initialize chat messages session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display welcome message if empty
if not st.session_state.messages:
    st.markdown(f"""
    <div style="background: rgba(139, 92, 246, 0.1); border: 1px solid rgba(139, 92, 246, 0.2); padding: 1.5rem; border-radius: 12px; margin-bottom: 2rem;">
        <h4>Welcome to GourmetAgent AI! 👋</h4>
        <p>I am your smart, premium restaurant assistant. Powered by <b>LangChain</b> and <b>Groq Llama-3.3-70B</b>, I combine our <b>Vector DB (RAG)</b> knowledge with <b>Live Web Search</b> to solve any restaurant query. Here's what you can ask me:</p>
        <ul>
            <li>📝 <b>Catering & Reservations:</b> Ask how to order, cancel, or modify reservations/catering.</li>
            <li>🔍 <b>Live Web Search:</b> Ask for real-world restaurant recommendations, reviews, or menus in any city.</li>
            <li>💳 <b>Store Policies:</b> Inquire about payment methods, safety guidelines, and allergens.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Render chat history
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        st.markdown(f"""
        <div class="chat-bubble chat-user">
            <b>👤 You</b><br/>{msg.content}
        </div>
        """, unsafe_allow_html=True)
    elif isinstance(msg, AIMessage):
        st.markdown(f"""
        <div class="chat-bubble chat-assistant">
            <b>🍽️ GourmetAgent AI</b><br/>{msg.content}
        </div>
        """, unsafe_allow_html=True)

# Handle user input (either from suggestion chips or text input)
user_query = st.chat_input("Ask GourmetAgent about restaurant policies, catering, or local recommendations...")

if selected_suggestion:
    user_query = selected_suggestion

if user_query:
    # Display user message
    st.markdown(f"""
    <div class="chat-bubble chat-user">
        <b>👤 You</b><br/>{user_query}
    </div>
    """, unsafe_allow_html=True)
    
    st.session_state.messages.append(HumanMessage(content=user_query))
    
    # Process and stream agent response
    with st.spinner("🕵️ GourmetAgent is searching knowledge bases and web sources..."):
        try:
            start_time = time.time()
            
            # Prepare full message history for langgraph agent
            input_messages = []
            for msg in st.session_state.messages:
                input_messages.append(msg)
                
            response = agent_executor.invoke({
                "messages": input_messages
            })
            duration = time.time() - start_time
            
            # The last message in the list returned contains the agent's output
            ai_content = response["messages"][-1].content
            
            # Display AI response
            st.markdown(f"""
            <div class="chat-bubble chat-assistant">
                <b>🍽️ GourmetAgent AI</b> <span style="font-size:0.75rem; color:#94a3b8; margin-left:10px;">(processed in {duration:.2f}s)</span><br/>
                {ai_content}
            </div>
            """, unsafe_allow_html=True)
            
            st.session_state.messages.append(AIMessage(content=ai_content))
            
        except Exception as e:
            import traceback
            with open("error_log.txt", "w") as f:
                traceback.print_exc(file=f)
            st.error(f"An error occurred while communicating with the agent: {e}")
            
    # Refresh to update the UI
    st.rerun()
