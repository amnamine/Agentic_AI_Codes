import streamlit as st
import time
import pandas as pd
from database import init_db, get_order, get_account
from agent import process_customer_query

# Page config
st.set_page_config(
    page_title="Aegis - AI Customer Support Agent",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Session State
init_db()
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am **Aegis**, your AI Customer Support Agent. How can I assist you with your orders, account, shipping address, or subscription today?"}
    ]
if "last_trace" not in st.session_state:
    st.session_state.last_trace = None

# Custom CSS for Premium Design
st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');
    
    /* Global styles */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Set custom dark luxury background */
    .stApp {
        background-color: #080810;
        background-image: radial-gradient(circle at 10% 20%, rgba(99, 102, 241, 0.05) 0%, transparent 40%),
                          radial-gradient(circle at 90% 80%, rgba(6, 182, 212, 0.05) 0%, transparent 40%);
        color: #E2E8F0;
    }
    
    /* Custom headers */
    h1, h2, h3 {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    
    .gradient-text {
        background: linear-gradient(135deg, #8B5CF6 0%, #06B6D4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }
    
    /* Glassmorphism card container */
    .glass-card {
        background: rgba(18, 18, 36, 0.7);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
    }
    
    .glass-card-header {
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        padding-bottom: 10px;
        margin-bottom: 15px;
        font-size: 1.15rem;
        font-weight: 600;
        color: #06B6D4;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Badges */
    .status-badge {
        padding: 3px 8px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-block;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .status-delivered {
        background: rgba(16, 185, 129, 0.15);
        color: #10B981;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .status-shipped {
        background: rgba(6, 182, 212, 0.15);
        color: #06B6D4;
        border: 1px solid rgba(6, 182, 212, 0.3);
    }
    .status-pending {
        background: rgba(245, 158, 11, 0.15);
        color: #F59E0B;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }
    .status-cancelled {
        background: rgba(244, 63, 94, 0.15);
        color: #F43F5E;
        border: 1px solid rgba(244, 63, 94, 0.3);
    }
    
    /* Custom chat bubble animations and styles */
    .chat-bubble-user {
        background: linear-gradient(135deg, #6366F1 0%, #4F46E5 100%);
        color: #FFFFFF;
        border-radius: 18px 18px 2px 18px;
        padding: 12px 16px;
        margin: 8px 0;
        align-self: flex-end;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.2);
        max-width: 80%;
        animation: slideInRight 0.3s ease-out;
    }
    
    .chat-bubble-agent {
        background: rgba(25, 25, 45, 0.85);
        border: 1px solid rgba(255, 255, 255, 0.08);
        color: #F1F5F9;
        border-radius: 18px 18px 18px 2px;
        padding: 12px 16px;
        margin: 8px 0;
        align-self: flex-start;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        max-width: 80%;
        animation: slideInLeft 0.3s ease-out;
    }

    @keyframes slideInRight {
        from { opacity: 0; transform: translateY(10px) translateX(10px); }
        to { opacity: 1; transform: translateY(0) translateX(0); }
    }
    @keyframes slideInLeft {
        from { opacity: 0; transform: translateY(10px) translateX(-10px); }
        to { opacity: 1; transform: translateY(0) translateX(0); }
    }
    
    /* Hide default streamlit elements for custom feel */
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# App Header
st.html("""
<div style='text-align: center; margin-bottom: 25px;'>
    <h1 style='font-size: 2.8rem; margin-bottom: 5px;'><span class='gradient-text'>🛡️ AEGIS</span></h1>
    <p style='color: #94A3B8; font-size: 1.1rem; max-width: 800px; margin: 0 auto;'>
        Autonomous Customer Support Agent powered by <b>Llama 3.3 (70B)</b> and <b>Agentic RAG</b>.
        Interact with the agent, inspect its inner thoughts, and watch the mock database update in real time.
    </p>
</div>
""")

# Layout columns
col_db, col_chat, col_mind = st.columns([3, 5, 4])

# ==========================================
# COLUMN 1: MOCK DATABASE EXPLORER
# ==========================================
with col_db:
    st.html("""
<div class='glass-card-header'>
    💾 MOCK DATABASE MONITOR
</div>
""")
    
    # Calculate metrics dynamically
    total_orders = len(st.session_state.orders)
    cancelled_orders = sum(1 for o in st.session_state.orders.values() if o["status"] == "Cancelled")
    refunded_orders = sum(1 for o in st.session_state.orders.values() if o["refund_status"] == "Refunded")
    
    st.html(f"""
<div style='display: flex; gap: 8px; margin-bottom: 15px;'>
    <div class='glass-card' style='flex: 1; padding: 10px; text-align: center; margin-bottom: 0; border-left: 3px solid #8B5CF6;'>
        <div style='font-size: 0.75rem; color: #94A3B8;'>Orders</div>
        <div style='font-size: 1.2rem; font-weight: 700; color: #C084FC;'>{total_orders}</div>
    </div>
    <div class='glass-card' style='flex: 1; padding: 10px; text-align: center; margin-bottom: 0; border-left: 3px solid #F43F5E;'>
        <div style='font-size: 0.75rem; color: #94A3B8;'>Cancelled</div>
        <div style='font-size: 1.2rem; font-weight: 700; color: #FB7185;'>{cancelled_orders}</div>
    </div>
    <div class='glass-card' style='flex: 1; padding: 10px; text-align: center; margin-bottom: 0; border-left: 3px solid #06B6D4;'>
        <div style='font-size: 0.75rem; color: #94A3B8;'>Refunds</div>
        <div style='font-size: 1.2rem; font-weight: 700; color: #22D3EE;'>{refunded_orders}</div>
    </div>
</div>
""")
    
    db_tab = st.radio("Select View:", ["Orders", "Accounts & Subscriptions", "Control Panel"], horizontal=True)
    
    if db_tab == "Orders":
        st.markdown("<h4 style='color: #E2E8F0; font-size: 0.95rem; margin-bottom: 10px;'>Active Orders in Database</h4>", unsafe_allow_html=True)
        for order_id, order in st.session_state.orders.items():
            status_class = f"status-badge status-{order['status'].lower()}"
            refund_badge = ""
            if order["refund_status"] == "Refunded":
                refund_badge = "<span class='status-badge' style='background: rgba(239,68,68,0.2); color: #EF4444; border: 1px solid rgba(239,68,68,0.4); margin-left: 5px;'>REFUNDED</span>"
                
            st.html(f"""
<div class='glass-card' style='padding: 12px; margin-bottom: 10px; border-left: 4px solid #8B5CF6;'>
    <div style='display: flex; justify-content: space-between; align-items: center;'>
        <strong>{order_id}</strong>
        <div>
            <span class='{status_class}'>{order['status']}</span>
            {refund_badge}
        </div>
    </div>
    <div style='font-size: 0.85rem; color: #94A3B8; margin-top: 8px;'>
        <div><b>Items:</b> {order['items']}</div>
        <div><b>Shipping Address:</b> {order['shipping_address']}</div>
        <div><b>Tracking:</b> {order['tracking_number']}</div>
        <div><b>Total:</b> ${order['total']:.2f}</div>
    </div>
</div>
""")
            
    elif db_tab == "Accounts & Subscriptions":
        st.markdown("<h4 style='color: #E2E8F0; font-size: 0.95rem; margin-bottom: 10px;'>Registered User Accounts</h4>", unsafe_allow_html=True)
        for email, acc in st.session_state.accounts.items():
            news_badge = "<span class='status-badge status-delivered'>Subscribed to Newsletter</span>" if acc["newsletter"] else "<span class='status-badge status-cancelled'>No Newsletter</span>"
            st.html(f"""
<div class='glass-card' style='padding: 12px; margin-bottom: 10px; border-left: 4px solid #06B6D4;'>
    <div style='display: flex; justify-content: space-between; align-items: center;'>
        <strong>{acc['name']}</strong>
        <span class='status-badge status-shipped'>{acc['status']}</span>
    </div>
    <div style='font-size: 0.85rem; color: #94A3B8; margin-top: 8px;'>
        <div><b>Email:</b> {acc['email']}</div>
        <div><b>Created At:</b> {acc['created_at']}</div>
        <div style='margin-top: 5px;'>{news_badge}</div>
    </div>
</div>
""")
            
    elif db_tab == "Control Panel":
        st.markdown("<h4 style='color: #E2E8F0; font-size: 0.95rem; margin-bottom: 10px;'>Database Seeding & Controls</h4>", unsafe_allow_html=True)
        if st.button("Reset Database to Default"):
            from database import DEFAULT_ACCOUNTS, DEFAULT_ORDERS
            st.session_state.accounts = DEFAULT_ACCOUNTS.copy()
            st.session_state.orders = DEFAULT_ORDERS.copy()
            st.success("Database restored successfully!")
            time.sleep(0.5)
            st.rerun()
            
        st.markdown("""
        <div style='font-size: 0.85rem; color: #94A3B8; margin-top: 15px; border-top: 1px solid rgba(255,255,255,0.08); padding-top: 10px;'>
            <b>How to test with the suggestions:</b> Click any of the quick-test buttons in the center panel to auto-fill the user query.
        </div>
        """, unsafe_allow_html=True)
        
        # Form to add custom orders for testing
        st.markdown("<h4 style='color: #E2E8F0; font-size: 0.95rem; margin-top: 15px; margin-bottom: 10px;'>➕ Create Custom Order</h4>", unsafe_allow_html=True)
        with st.form("custom_order_form", clear_on_submit=True):
            new_id = st.text_input("Order Number", placeholder="e.g. ORD-99999")
            new_items = st.text_input("Items Description", placeholder="e.g. 1x Wireless Charger")
            new_total = st.number_input("Total Amount ($)", min_value=0.0, value=25.00, step=0.01)
            new_addr = st.text_input("Shipping Address", placeholder="e.g. 123 Pine St, Seattle")
            new_status = st.selectbox("Order Status", ["Pending", "Shipped", "Delivered"])
            submit_order = st.form_submit_button("Add Order to DB")
            
            if submit_order:
                if not new_id or not new_items or not new_addr:
                    st.error("Please fill out all order details.")
                else:
                    import random
                    new_id_clean = new_id.strip().upper()
                    if not new_id_clean.startswith("ORD-"):
                        new_id_clean = f"ORD-{new_id_clean}"
                        
                    st.session_state.orders[new_id_clean] = {
                        "order_number": new_id_clean,
                        "email": "customer@example.com",
                        "items": new_items,
                        "total": float(new_total),
                        "status": new_status,
                        "shipping_address": new_addr,
                        "tracking_number": f"TRK-{random.randint(10000000, 99999999)}",
                        "cancellation_fee": 0.0,
                        "refund_status": "None"
                    }
                    st.success(f"Order {new_id_clean} successfully added!")
                    time.sleep(0.5)
                    st.rerun()


# ==========================================
# COLUMN 2: CHATBOT CONTAINER
# ==========================================
with col_chat:
    st.html("""
<div class='glass-card-header'>
    💬 CUSTOMER SUPPORT CHAT
</div>
""")
    
    # Suggestion Chips / Buttons
    st.markdown("<div style='font-size: 0.85rem; color: #94A3B8; margin-top: 5px; margin-bottom: 5px;'>⚡ Quick Test Suggestions (Click to trigger):</div>", unsafe_allow_html=True)
    
    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        if st.button("❌ Cancel ORD-11502", key="sug_cancel", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": "I want to cancel order ORD-11502 please"})
            st.rerun()
    with col_s2:
        if st.button("📦 Track ORD-88231", key="sug_track", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": "Can you track order ORD-88231?"})
            st.rerun()
    with col_s3:
        if st.button("💳 Refund Policy", key="sug_refund", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": "What is your refund policy?"})
            st.rerun()

    col_s4, col_s5, col_s6 = st.columns(3)
    with col_s4:
        if st.button("📍 Change Address", key="sug_addr", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": "Change address of ORD-45691 to 742 Evergreen Terrace, Springfield"})
            st.rerun()
    with col_s5:
        if st.button("📧 Subscribe News", key="sug_sub", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": "Subscribe my email test@example.com to the newsletter"})
            st.rerun()
    with col_s6:
        if st.button("💸 Refund Order", key="sug_ref_req", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": "I want a refund for ORD-88231"})
            st.rerun()
    
    # Custom chat viewport
    chat_container = st.container(height=500)
    
    with chat_container:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.html(f"""
<div style='display: flex; justify-content: flex-end;'>
    <div class='chat-bubble-user'>
        {msg['content']}
    </div>
</div>
""")
            else:
                st.html(f"""
<div style='display: flex; justify-content: flex-start;'>
    <div class='chat-bubble-agent'>
        {msg['content']}
    </div>
</div>
""")
                
    # User Input
    if user_input := st.chat_input("Ask a question, track/cancel order, manage account..."):
        # Display user message instantly
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.rerun()

# Trigger processing if the last message is from the user
if st.session_state.messages[-1]["role"] == "user":
    user_query = st.session_state.messages[-1]["content"]
    
    # Process
    with st.spinner("Aegis thinking..."):
        response, trace = process_customer_query(user_query, st.session_state.messages[:-1])
        
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.session_state.last_trace = trace
    st.rerun()


# ==========================================
# COLUMN 3: AGENT MIND INSPECTOR
# ==========================================
with col_mind:
    st.html("""
<div class='glass-card-header'>
    🧠 AGENT MIND INSPECTOR
</div>
""")
    
    if st.session_state.last_trace is None:
        st.info("Ask the chatbot a question to inspect the Agent's internal decision-making steps, intent classification, and RAG context.")
    else:
        trace = st.session_state.last_trace
        
        # 1. Thought process / reasoning
        st.html(f"""
<div class='glass-card' style='background: rgba(139, 92, 246, 0.08); border-color: rgba(139, 92, 246, 0.25);'>
    <div style='font-size: 0.85rem; font-weight: 600; color: #8B5CF6; margin-bottom: 5px; text-transform: uppercase;'>Inner Reasoning</div>
    <div style='font-size: 0.9rem; font-style: italic; color: #E2E8F0;'>"{trace['reasoning']}"</div>
</div>
""")
        
        # 2. Classification details
        st.html(f"""
<div class='glass-card' style='background: rgba(6, 182, 212, 0.08); border-color: rgba(6, 182, 212, 0.25);'>
    <div style='font-size: 0.85rem; font-weight: 600; color: #06B6D4; margin-bottom: 5px; text-transform: uppercase;'>Classification Details</div>
    <div style='font-size: 0.9rem;'>
        <b>Category:</b> <span class='status-badge status-shipped'>{trace['category']}</span><br/>
        <div style='margin-top:5px;'><b>Intent:</b> <code>{trace['intent']}</code></div>
    </div>
</div>
""")
        
        # 3. Extracted Entities
        entities_html = ""
        if trace["entities"]:
            for k, v in trace["entities"].items():
                if v:
                    entities_html += f"<div><b>{k}:</b> <code>{v}</code></div>"
        if not entities_html:
            entities_html = "<div style='color: #64748B;'>No entities extracted.</div>"
            
        st.html(f"""
<div class='glass-card' style='background: rgba(16, 185, 129, 0.08); border-color: rgba(16, 185, 129, 0.25);'>
    <div style='font-size: 0.85rem; font-weight: 600; color: #10B981; margin-bottom: 5px; text-transform: uppercase;'>Extracted Entities</div>
    <div style='font-size: 0.9rem;'>{entities_html}</div>
</div>
""")
        
        # 4. Tools and Actions
        st.html(f"""
<div class='glass-card' style='background: rgba(245, 158, 11, 0.08); border-color: rgba(245, 158, 11, 0.25);'>
    <div style='font-size: 0.85rem; font-weight: 600; color: #F59E0B; margin-bottom: 5px; text-transform: uppercase;'>Tool Execution Trace</div>
    <div style='font-size: 0.85rem; color: #E2E8F0; font-family: monospace;'>{trace['tool_status']}</div>
</div>
""")
        
        # 5. RAG Retrieval Content
        with st.expander("📄 RAG Retrieved Context (CSV references)", expanded=False):
            if trace["rag_context"]:
                st.markdown(trace["rag_context"])
            else:
                st.write("No matching references fetched.")
