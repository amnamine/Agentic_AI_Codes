import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

import streamlit as st
import pandas as pd
import numpy as np
import os
import time
import re
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ----------------- CONFIGURATION & PAGE INITIALIZATION -----------------
st.set_page_config(
    page_title="LuxeCart | Premium AI E-commerce & Agentic RAG",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif;
        font-weight: 600;
        letter-spacing: -0.5px;
    }

    .stApp {
        background: linear-gradient(135deg, #070913 0%, #0c0f24 100%);
        color: #e2e8f0;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: rgba(10, 12, 26, 0.96);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(15px);
    }

    /* Titles */
    .brand-title {
        background: linear-gradient(90deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.8rem;
        margin-bottom: 0.1rem;
        letter-spacing: -1px;
    }
    .brand-subtitle {
        color: #94a3b8;
        font-size: 1rem;
        margin-bottom: 1.5rem;
        font-weight: 300;
    }

    /* Product Cards */
    .product-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
        gap: 20px;
        margin-bottom: 2rem;
    }
    .product-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 16px;
        padding: 16px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        position: relative;
        overflow: hidden;
    }
    .product-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; height: 3px;
        background: linear-gradient(90deg, #6366f1, #a855f7);
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    .product-card:hover {
        transform: translateY(-5px);
        border-color: rgba(99, 102, 241, 0.4);
        background: rgba(255, 255, 255, 0.05);
        box-shadow: 0 10px 30px rgba(99, 102, 241, 0.08);
    }
    .product-card:hover::before {
        opacity: 1;
    }
    .product-tag {
        background: rgba(99, 102, 241, 0.15);
        color: #818cf8;
        font-size: 0.7rem;
        font-weight: 700;
        padding: 4px 8px;
        border-radius: 6px;
        align-self: flex-start;
        text-transform: uppercase;
        margin-bottom: 10px;
    }
    .product-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #f8fafc;
        margin-bottom: 8px;
    }
    .product-price {
        font-size: 1.2rem;
        font-weight: 700;
        color: #34d399;
        margin-top: auto;
    }

    /* Glassmorphism panel */
    .glass-panel {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 20px;
        backdrop-filter: blur(10px);
        margin-bottom: 1.5rem;
    }

    /* Agentic RAG Step badges */
    .step-badge {
        display: inline-flex;
        align-items: center;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: bold;
        margin-right: 8px;
        margin-bottom: 8px;
    }
    .step-router { background: rgba(239, 68, 68, 0.15); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.3); }
    .step-retriever { background: rgba(59, 130, 246, 0.15); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.3); }
    .step-llm { background: rgba(168, 85, 247, 0.15); color: #c084fc; border: 1px solid rgba(168, 85, 247, 0.3); }

    /* Custom Chat bubbles */
    .chat-container {
        display: flex;
        flex-direction: column;
        gap: 14px;
        margin-bottom: 2rem;
    }
    .chat-bubble {
        display: flex;
        padding: 14px 18px;
        border-radius: 16px;
        max-width: 85%;
        line-height: 1.5;
        font-size: 0.95rem;
        animation: slideUp 0.3s ease;
    }
    @keyframes slideUp {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .chat-bubble.user {
        background: linear-gradient(135deg, #4f46e5 0%, #3730a3 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        align-self: flex-end;
        color: #ffffff;
        border-bottom-right-radius: 4px;
        box-shadow: 0 4px 15px rgba(79, 70, 229, 0.3);
    }
    .chat-bubble.assistant {
        background: rgba(17, 24, 39, 0.8);
        border: 1px solid rgba(255, 255, 255, 0.05);
        align-self: flex-start;
        color: #e2e8f0;
        border-bottom-left-radius: 4px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    .bubble-avatar {
        font-size: 1.25rem;
        margin-right: 10px;
        display: flex;
        align-items: center;
    }
    .bubble-content {
        flex: 1;
    }
    
    /* Scrollbars */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    ::-webkit-scrollbar-track {
        background: rgba(0,0,0,0.1);
    }
    ::-webkit-scrollbar-thumb {
        background: rgba(255,255,255,0.08);
        border-radius: 3px;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- VECTOR STORE & CSV LOADER -----------------

CSV_PATH = r"d:\AMINE2\COURS FAC\LEARNING\YOUTUBER\Codes\AI_Agent_AgenticRAG\ecommerce.csv"
INDEX_DIR = r"d:\AMINE2\COURS FAC\LEARNING\YOUTUBER\Codes\AI_Agent_AgenticRAG\faiss_index"

@st.cache_resource(show_spinner="Initializing Vector DB & E-Commerce catalog...")
def init_rag_knowledge_base():
    # 1. Load subset of the CSV to build the FAQ & intent templates
    # We sample a subset to keep local indexing fast & fully featured
    df = pd.read_csv(CSV_PATH)
    
    # We group by category and intent to have a dense representation of all actions
    df_clean = df.dropna(subset=['instruction', 'response']).drop_duplicates(subset=['instruction'])
    
    # Take a sample for standard customer queries
    df_sample = df_clean.sample(n=min(3000, len(df_clean)), random_state=42)
    
    # Build Langchain Documents
    from langchain_core.documents import Document
    documents = []
    for _, row in df_sample.iterrows():
        doc = Document(
            page_content=str(row['instruction']),
            metadata={
                'intent': str(row['intent']),
                'category': str(row['category']),
                'response': str(row['response'])
            }
        )
        documents.append(doc)
        
    # Generate HuggingFace embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    
    # Create FAISS index
    vectorstore = FAISS.from_documents(documents, embeddings)
    
    # Create simulated storefront items based on unique category keywords
    store_categories = df_clean['category'].unique()
    products = [
        {"id": 1, "name": "Ultra-Fit Comfort Shoes", "category": "SHIPPING", "price": 89.99, "tags": "Best Seller"},
        {"id": 2, "name": "Carbon Fibre Premium Wallet", "category": "PAYMENT", "price": 45.00, "tags": "Security"},
        {"id": 3, "name": "HyperCharge Wireless Pad", "category": "ORDER", "price": 29.99, "tags": "New Release"},
        {"id": 4, "name": "AeroFoam Sleeping Pillow", "category": "REFUND", "price": 59.99, "tags": "Top Rated"},
        {"id": 5, "name": "Quantum Noise Cancelling Headphones", "category": "CANCEL", "price": 199.99, "tags": "Premium"},
        {"id": 6, "name": "EvoGrip Mechanical Keyboard", "category": "CONTACT", "price": 120.00, "tags": "Hot Choice"}
    ]
    
    return vectorstore, products

vectorstore, catalog_products = init_rag_knowledge_base()

# ----------------- SESSION STATE -----------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "cart" not in st.session_state:
    st.session_state.cart = []
if "agent_steps" not in st.session_state:
    st.session_state.agent_steps = {}

# ----------------- AGENTIC RAG PIPELINE -----------------

def agentic_rag_pipeline(query: str):
    # STEP 1: Intent Routing
    # Evaluate context and determine where to route the request
    # Router helps decide if it should perform a refund search, add items, etc.
    st.session_state.agent_steps = {}
    
    router_prompt = f"""You are an expert E-Commerce router agent. Analyze the customer's query and classify it into one of these paths:
- CATALOG: Asking about specific store products, item prices, or looking to buy.
- SUPPORT: Standard customer issues (refund policy, order status, shipping questions, payment issues, cancel request, cart help, account setup).
- OTHER: Off-topic questions, programming, general knowledge.

Customer Query: "{query}"

Output ONLY the path name in uppercase: CATALOG, SUPPORT, or OTHER. Do not output anything else.
"""
    
    # Initialize Groq client
    llm = ChatGroq(
        groq_api_key="",
        model_name="llama-3.3-70b-versatile",
        temperature=0.0
    )
    
    # Route query
    try:
        route_decision = llm.invoke(router_prompt).content.strip()
    except Exception as e:
        route_decision = "SUPPORT"  # Fallback to general support
    
    st.session_state.agent_steps["Route Decision"] = route_decision
    
    # STEP 2: Adaptive Retrieval
    retrieved_context = ""
    retrieved_records = []
    
    if route_decision == "SUPPORT":
        # Search FAISS vector store for semantic context templates
        results = vectorstore.similarity_search_with_score(query, k=2)
        for doc, distance in results:
            sim = 1.0 / (1.0 + distance)
            retrieved_records.append({
                'intent': doc.metadata.get('intent'),
                'category': doc.metadata.get('category'),
                'response': doc.metadata.get('response'),
                'similarity': sim
            })
            
        # Format the retrieved response templates
        retrieved_context = "\n\n".join([
            f"Intent: {rec['intent']}\nResponse Template: {rec['response']}"
            for rec in retrieved_records
        ])
        st.session_state.agent_steps["Retrieved Records"] = retrieved_records
        
    elif route_decision == "CATALOG":
        # Search E-commerce products in the catalog
        matching_products = [
            p for p in catalog_products 
            if any(word in p['name'].lower() or word in p['category'].lower() for word in query.lower().split())
        ]
        if not matching_products:
            matching_products = catalog_products[:3]  # Return standard catalog if no direct match
            
        retrieved_context = "Available Products in Catalog:\n" + "\n".join([
            f"- {p['name']} ({p['category']}): ${p['price']} [Tags: {p['tags']}]"
            for p in matching_products
        ])
        st.session_state.agent_steps["Retrieved Products"] = matching_products

    # STEP 3: LLM Generation (Reasoning Step)
    system_instruction = """You are LuxeCart's AI E-Commerce Shopping & Support assistant.
Your goal is to answer the customer's query using the retrieved context (which contains product catalog details or support templates).

Rules:
- Be extremely polite, professional, and helpful.
- Replace placeholders (e.g. {{Product Catalog}}, {{Add to Cart}}, {{Checkout}}, {{Customer Support Phone Number}}) with realistic, beautifully styled values.
- If you're suggesting a product, list its name and price clearly.
- If the customer's question is OUTSIDE shopping and e-commerce support, politely remind them that you are specialized in LuxeCart store products and operations.
"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_instruction),
        ("user", "Retrieved Context:\n{context}\n\nCustomer Query: {query}\n\nAssistant Response:")
    ])
    
    chain = prompt | llm | StrOutputParser()
    
    try:
        response = chain.invoke({"context": retrieved_context if retrieved_context else "No specific templates matched.", "query": query})
    except Exception as err:
        response = f"⚠️ Groq API Error: {err}"
        
    return response

# ----------------- MAIN LAYOUT -----------------

# Header
st.markdown("<div class='brand-title'>LuxeCart</div>", unsafe_allow_html=True)
st.markdown("<div class='brand-subtitle'>Agentic RAG E-Commerce Chatbot Platform</div>", unsafe_allow_html=True)

# Main layout split
col_products, col_chat = st.columns([1.1, 1.3])

# left side: E-commerce Storefront
with col_products:
    st.markdown("### 🛍️ Featured Storefront Products")
    st.markdown("<div class='product-grid'>", unsafe_allow_html=True)
    
    # Display products as elegant cards
    cols = st.columns(2)
    for idx, p in enumerate(catalog_products):
        with cols[idx % 2]:
            st.markdown(f"""
            <div class='product-card'>
                <span class='product-tag'>{p['tags']}</span>
                <div class='product-title'>{p['name']}</div>
                <div style='color: #94a3b8; font-size: 0.8rem; margin-bottom: 12px;'>Category: {p['category']}</div>
                <div class='product-price'>${p['price']:.2f}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Simple purchase/add actions
            if st.button(f"🛒 Add {p['name'].split()[0]} to Cart", key=f"add_{p['id']}", use_container_width=True):
                st.session_state.cart.append(p)
                st.toast(f"Added {p['name']} to your Cart! 🎉")

    # Shopping Cart Status
    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
    with st.expander("💼 Your Shopping Cart", expanded=True):
        if st.session_state.cart:
            total_price = 0.0
            for item in st.session_state.cart:
                st.markdown(f"- **{item['name']}** - ${item['price']:.2f}")
                total_price += item['price']
            st.markdown(f"**Total: ${total_price:.2f}**")
            if st.button("Clear Cart", use_container_width=True):
                st.session_state.cart = []
                st.rerun()
        else:
            st.info("Your shopping cart is currently empty.")

# right side: Chatbot
with col_chat:
    st.markdown("### 💬 LuxeCart Assistant")
    
    # Custom Chat logs wrapper
    chat_container_html = "<div class='chat-container'>"
    for msg in st.session_state.messages:
        role = msg["role"]
        avatar = "👤" if role == "user" else "✨"
        chat_container_html += f"<div class='chat-bubble {role}'><div class='bubble-avatar'>{avatar}</div><div class='bubble-content'>{msg['content']}</div></div>"
    chat_container_html += "</div>"
    st.markdown(chat_container_html, unsafe_allow_html=True)

    # FAQ Suggestions
    if not st.session_state.messages:
        st.markdown("<div style='margin-top: 0.5rem;'></div>", unsafe_allow_html=True)
        st.markdown("##### 💡 Suggested Questions")
        faq_cols = st.columns(2)
        suggestions = [
            ("💳 How do I request a refund?", "I want to request a refund, how can I do that?"),
            ("🚚 What is the shipping time?", "What are the shipping methods and delivery timings?"),
            ("🛒 Add items to basket", "How can I add an item to my shopping cart?"),
            ("🔒 Can I update account details?", "I need help updating my personal private account data.")
        ]
        for idx, (label, query) in enumerate(suggestions):
            with faq_cols[idx % 2]:
                if st.button(label, key=f"faq_{idx}", use_container_width=True):
                    st.session_state.messages.append({"role": "user", "content": query})
                    st.rerun()

    # Chat Input
    user_input = st.chat_input("Ask about our products, refund policies, cart, or order support...")
    
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.rerun()

    # Trigger response generation
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        last_query = st.session_state.messages[-1]["content"]
        
        with st.spinner("Agentic RAG routing & retrieving..."):
            response = agentic_rag_pipeline(last_query)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()

    # Agentic RAG Step Inspector
    if st.session_state.agent_steps:
        st.markdown("<hr style='border-color: rgba(255,255,255,0.05);'>", unsafe_allow_html=True)
        with st.expander("🛠️ Agentic RAG Execution Steps", expanded=True):
            route = st.session_state.agent_steps.get("Route Decision")
            st.markdown(f"<span class='step-badge step-router'>Step 1: Router</span> Classifying query -> **{route}**", unsafe_allow_html=True)
            
            if route == "SUPPORT" and "Retrieved Records" in st.session_state.agent_steps:
                st.markdown(f"<span class='step-badge step-retriever'>Step 2: Retriever</span> Retrieved closest FAQ matches:", unsafe_allow_html=True)
                for rec in st.session_state.agent_steps["Retrieved Records"]:
                    st.markdown(f"- **Intent**: {rec['intent']} | **Similarity**: {rec['similarity']:.3f}")
                    st.markdown(f"  *Template*: _{rec['response'][:100]}..._")
            elif route == "CATALOG" and "Retrieved Products" in st.session_state.agent_steps:
                st.markdown(f"<span class='step-badge step-retriever'>Step 2: Retriever</span> Retrieved closest Products:", unsafe_allow_html=True)
                for p in st.session_state.agent_steps["Retrieved Products"]:
                    st.markdown(f"- **{p['name']}** ({p['category']}) - ${p['price']}")
            else:
                st.markdown(f"<span class='step-badge step-retriever'>Step 2: Retriever</span> Generic RAG retrieval bypass (Other request)", unsafe_allow_html=True)
                
            st.markdown(f"<span class='step-badge step-llm'>Step 3: LLM Reasoner</span> Prompting Llama-3.3-70b-versatile with context for final synthesis.", unsafe_allow_html=True)
