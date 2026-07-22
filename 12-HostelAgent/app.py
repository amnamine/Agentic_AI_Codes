import streamlit as st
import pandas as pd
import numpy as np
import faiss
import pickle
import os
import random
from sentence_transformers import SentenceTransformer
from groq import Groq

# Page Config
st.set_page_config(
    page_title="CozyBot - Premium Hostel Companion",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Luxurious Cyberpunk/Glassmorphic Theme)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Inter:wght@300;400;500;600;700&display=swap');

/* Apply primary font to everything */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: #e2e8f0;
    background-color: #0d0e12;
}

/* Custom titles styling */
.main-title {
    font-family: 'Outfit', sans-serif;
    font-weight: 800;
    font-size: 3rem;
    background: linear-gradient(135deg, #a78bfa 0%, #ec4899 50%, #f43f5e 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 5px;
    text-shadow: 0 0 30px rgba(167, 139, 250, 0.2);
}

.subtitle {
    font-family: 'Outfit', sans-serif;
    font-weight: 400;
    font-size: 1.1rem;
    color: #94a3b8;
    margin-bottom: 25px;
}

/* Glassmorphic Sidebar styling */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(17, 24, 39, 0.95) 0%, rgba(13, 14, 18, 0.98) 100%) !important;
    border-right: 1px solid rgba(167, 139, 250, 0.15) !important;
    box-shadow: 0 4px 30px rgba(0, 0, 0, 0.5);
    backdrop-filter: blur(10px);
}

/* Custom containers & cards */
.custom-card {
    background: rgba(30, 41, 59, 0.45);
    border: 1px solid rgba(167, 139, 250, 0.15);
    border-radius: 16px;
    padding: 20px;
    margin-bottom: 15px;
    backdrop-filter: blur(12px);
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
    transition: transform 0.3s ease, border-color 0.3s ease;
}

.custom-card:hover {
    transform: translateY(-2px);
    border-color: rgba(167, 139, 250, 0.35);
    box-shadow: 0 12px 40px 0 rgba(167, 139, 250, 0.1);
}

/* Chat Message Styling override */
div[data-testid="stChatMessage"] {
    background-color: rgba(22, 28, 45, 0.5) !important;
    border: 1px solid rgba(167, 139, 250, 0.1);
    border-radius: 16px;
    padding: 16px;
    margin-bottom: 12px;
    transition: transform 0.2s ease;
}

div[data-testid="stChatMessage"]:hover {
    transform: translateY(-1px);
    border-color: rgba(167, 139, 250, 0.2);
}

/* Chat input bar customization */
[data-testid="stChatInput"] {
    border-radius: 14px !important;
    border: 1px solid rgba(167, 139, 250, 0.2) !important;
    background-color: rgba(17, 24, 39, 0.8) !important;
}

/* Suggestion Buttons layout */
.suggestion-btn {
    border: 1px solid rgba(167, 139, 250, 0.25) !important;
    background-color: rgba(167, 139, 250, 0.05) !important;
    color: #c084fc !important;
    border-radius: 20px !important;
    font-weight: 500 !important;
    padding: 6px 16px !important;
    transition: all 0.3s ease !important;
}

.suggestion-btn:hover {
    background-color: rgba(167, 139, 250, 0.15) !important;
    border-color: #a78bfa !important;
    color: #fff !important;
    transform: scale(1.02);
}

/* Micro-animations */
@keyframes pulse {
    0% { transform: scale(1); box-shadow: 0 0 0 0 rgba(167, 139, 250, 0.4); }
    70% { transform: scale(1.02); box-shadow: 0 0 0 10px rgba(167, 139, 250, 0); }
    100% { transform: scale(1); box-shadow: 0 0 0 0 rgba(167, 139, 250, 0); }
}

.pulse-badge {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 12px;
    background: rgba(52, 211, 153, 0.1);
    color: #34d399;
    font-size: 0.8rem;
    font-weight: 600;
    border: 1px solid rgba(52, 211, 153, 0.3);
    margin-bottom: 10px;
}

</style>
""", unsafe_allow_html=True)

# Placeholder replacement dictionary to turn generic template variables into cute Cozy Hostel specific details
PLACEHOLDER_REPLACEMENTS = {
    "WEBSITE_URL": "https://antigravity-cozy-hostel.com",
    "SUPPORT_NUMBER": "+34 932 456 789",
    "SUPPORT_INFORMATION": "stay@antigravity-cozy-hostel.com",
    "CITY": "Barcelona",
    "LOCATION": "Barcelona Eixample District, Spain",
    "DESTINATION_CITY": "Barcelona",
    "ORIGIN": "anywhere globally",
    "CANCELLATION_FEES_INFORMATION_TAB": "Cancellation Policy tab",
    "CANCELLATION_FEES_SECTION": "Cancellation and Refund Policies in your Account Dashboard",
    "CANCEL_RESERVATION_OPTION": "Cancel Booking option under My Bookings",
    "CHARGE_TYPE": "standard nightly rate",
    "CHECK_HOTEL_FACILITIES_INFORMATION_SECTION": "Amenities tab in the sidebar",
    "CHECK_MENU_OPTION": "Café Menu in the sidebar",
    "CHECK_OUT_OPTION": "Express Check-Out button in your guest profile",
    "CONTACT_US_SECTION": "Contact section of our site",
    "EVENTS_SECTION": "Upcoming Events tab in the sidebar",
    "HOTEL_FACILITIES_SECTION": "Amenities & Facilities list",
    "INVOICES_SECTION": "Invoices tab under Billing in your profile",
    "LEAVE_REVIEW_OPTION": "Leave a Review button in the Feedback area",
    "MANAGE_BOOKING_SECTION": "Manage Booking page",
    "MANAGE_RESERVATIONS_SECTION": "Manage Reservations dashboard",
    "MENU_SECTION": "Cozy Café & Tapas Bar Menu",
    "MODIFY_BOOKING_OPTION": "Modify Booking option in your reservations panel",
    "MODIFY_OPTION": "Modify details button",
    "MY_BOOKINGS_SECTION": "My Bookings section",
    "OFFERS_SECTION": "Special Deals & Packages",
    "PARKING_RESERVATION_SECTION": "Reserve Parking space under Booking Details",
    "POINTS_TYPE": "Loyalty Cozy Points",
    "REDEEM_POINTS_SECTION": "Redeem Cozy Points in the guest portal",
    "REFUND_SECTION": "Refunds and Disputes panel",
    "REQUEST_EARLY_CHECK-IN_OPTION": "Early Check-In request form",
    "REQUEST_REFUND_OPTION": "Request Refund button in billing",
    "RESERVATION_SECTION": "Reservation Center",
    "REVIEW_SECTION": "Hostel Reviews and Guestbook",
    "SAVE_BUTTON": "Save Changes button",
    "SEARCH_BUTTON": "Search Available Rooms button",
    "SEND_BUTTON": "Submit Request button",
    "SERVICES_SECTION": "Guest Services and Concierge Desk",
    "SHUTTLE_SERVICE": "Daily Airport Shuttle service",
    "SPEAK_WITH_HUMAN_AGENT_OPTION": "Speak with a Human Hostel Host",
}

def replace_placeholders(text):
    if not isinstance(text, str):
        return text
    for key, value in PLACEHOLDER_REPLACEMENTS.items():
        text = text.replace(f"{{{{{key}}}}}", value)
    return text

# Initialize Groq client
@st.cache_resource
def get_groq_client():
    # Hardcoded Groq API key requested by the user
    return Groq(api_key="")

# Load local FAISS vector index & SentenceTransformer model
@st.cache_resource
def load_rag_system():
    if not os.path.exists("vector_store.faiss") or not os.path.exists("vector_store_metadata.pkl"):
        return None, None, None
    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    index = faiss.read_index("vector_store.faiss")
    with open("vector_store_metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    return model, index, metadata

# Search local dataset using FAISS
def retrieve_relevant_contexts(query, model, index, metadata, k=3):
    if index is None or metadata is None or model is None:
        return []
    
    query_emb = model.encode([query]).astype('float32')
    distances, indices = index.search(query_emb, k)
    
    retrieved = []
    for idx in indices[0]:
        if idx < len(metadata):
            item = metadata[idx].copy()
            # Dynamic template rendering
            item['response'] = replace_placeholders(item['response'])
            retrieved.append(item)
    return retrieved

# System prompt outlining CozyBot behavior, hostel rates, and features
SYSTEM_PROMPT = """You are CozyBot, the friendly, charming virtual hostel host for "The Antigravity Cozy Hostel" located in Barcelona, Spain.
You are professional, welcoming, highly helpful, and conversational.

Your duties:
1. Assist guests with inquiries about booking, billing, check-in, checkout, cancellation policies, and facilities using the retrieved context snippets.
2. Recommend local Barcelona sightseeing spots (e.g. Park Güell, Sagrada Família, La Rambla, Gothic Quarter) and hostel activities.
3. Help users calculate booking rates and promote hostel extras when they express interest in booking:
   - Cozy Dorm Bed: $29/night
   - Private Cozy Double: $79/night
   - Luxury Penthouse Dorm: $49/night
   - Antigravity Theme Suite: $129/night
   - Extras: Breakfast ($8/day/guest), Parking ($15/day), Guided Pub Crawl ($20/person), Airport Shuttle ($25 flat)
4. Address the user with enthusiasm, using bullet points, emojis, and bold text to make responses easy to read.
5. If the retrieved snippets are relevant, integrate their instructions seamlessly into your friendly response. Do not mention raw codes or format schemas like JSON.

Retrieved Context Snippets from Hostel Operations Manual:
{context_text}
"""

# Load assets
model, index, metadata = load_rag_system()

# UI Layout: Header
st.markdown('<div class="main-title">🏨 CozyBot</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Your Luxurious AI Concierge for The Antigravity Cozy Hostel, Barcelona</div>', unsafe_allow_html=True)

# Main Grid layout: 8 columns chat, 4 columns hostel information and booking
chat_col, sidebar_col = st.columns([2, 1])

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "¡Hola! Welcome to **The Antigravity Cozy Hostel**! I'm **CozyBot**, your digital host. How can I make your stay in Barcelona unforgettable today? Ask me about bookings, check-in/out, local recommendations, café menus, or use the interactive simulator!"}
    ]

# If click suggestion chip
if "pending_input" not in st.session_state:
    st.session_state.pending_input = None

with sidebar_col:
    st.markdown("### 🌟 Hostel Dashboard")
    
    # Live stats
    st.markdown('<span class="pulse-badge">🟢 CozyBot Live</span>', unsafe_allow_html=True)
    if metadata:
        st.markdown(f"**Hostel Knowledge Base:** {len(metadata):,} articles active")
    else:
        st.markdown("**Hostel Knowledge Base:** Offline (Build Index first)")
    st.markdown("**LLM Engine:** Llama-3.3-70B-Versatile (Groq)")
    
    # Tab layout for sidebar info
    info_tabs = st.tabs(["🛏️ Rooms & Pricing", "🍳 Café & Events", "📅 Dynamic Booking Simulator"])
    
    with info_tabs[0]:
        st.markdown("""
        <div class="custom-card">
            <h4>🛏️ Room Categories & Nightly Rates</h4>
            <hr style="margin: 8px 0; border: 0.5px solid rgba(255,255,255,0.1)">
            <p><b>1. Cozy Dorm Bed</b> - <i>$29 / night</i><br>Shared 6-bed clean room, individual privacy curtains, USB ports, lockable drawer.</p>
            <p><b>2. Private Cozy Double</b> - <i>$79 / night</i><br>Comfortable double bed, en-suite private bathroom, smart TV, quiet balcony.</p>
            <p><b>3. Luxury Penthouse Dorm</b> - <i>$49 / night</i><br>Rooftop-level 4-bed dorm, lounge access, stunning skyline views.</p>
            <p><b>4. Antigravity Theme Suite</b> - <i>$129 / night</i><br>Premium cyberpunk capsule, magnetic floating bed, galaxy ceiling projector, mini-fridge.</p>
        </div>
        
        <div class="custom-card">
            <h4>🎯 Premium Amenities</h4>
            <ul>
                <li>Free High-Speed Wi-Fi (500 Mbps)</li>
                <li>Rooftop Terrace & Swimming Pool</li>
                <li>24/7 Front Desk Support</li>
                <li>Luggage Storage & Locker Facility</li>
                <li>Bicycle Rental & Local Metro Pass Access</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    with info_tabs[1]:
        st.markdown("""
        <div class="custom-card">
            <h4>🍳 Cozy Café & Tapas Bar Menu</h4>
            <hr style="margin: 8px 0; border: 0.5px solid rgba(255,255,255,0.1)">
            <ul>
                <li><b>Churros con Chocolate</b> - $4.50</li>
                <li><b>Patatas Bravas (Chef Special)</b> - $6.00</li>
                <li><b>Spanish Sangria Pitcher</b> - $14.00</li>
                <li><b>Café con Leche / Espresso</b> - $2.50</li>
                <li><b>Antigravity Full Breakfast</b> - $8.00</li>
            </ul>
        </div>
        
        <div class="custom-card">
            <h4>🎉 Weekly Guest Events</h4>
            <hr style="margin: 8px 0; border: 0.5px solid rgba(255,255,255,0.1)">
            <ul>
                <li><b>Monday:</b> Free Paella & Sangria Rooftop Dinner</li>
                <li><b>Wednesday:</b> Sunset Acoustic Guitar Sessions</li>
                <li><b>Friday:</b> Barcelona Gothic Quarter Pub Crawl ($20)</li>
                <li><b>Sunday:</b> Beach Volleyball & Picnic Match</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    with info_tabs[2]:
        st.markdown("#### 📅 Quote Calculator")
        room_choice = st.selectbox(
            "Select Room Type",
            ["Cozy Dorm Bed ($29/n)", "Private Cozy Double ($79/n)", "Luxury Penthouse Dorm ($49/n)", "Antigravity Theme Suite ($129/n)"]
        )
        nights = st.number_input("Number of Nights", min_value=1, max_value=30, value=2)
        guests = st.number_input("Number of Guests", min_value=1, max_value=8, value=1)
        
        st.markdown("**Select Extras:**")
        add_breakfast = st.checkbox("🍳 Daily Breakfast Buffet (+$8/day/guest)")
        add_parking = st.checkbox("🚗 Parking Spot Reservation (+$15/day)")
        add_crawl = st.checkbox("🍹 Guided Pub Crawl & Tour (+$20/guest)")
        add_shuttle = st.checkbox("🚐 Airport Shuttle Transfer (+$25 flat)")
        
        # Calculate pricing
        room_rates = {
            "Cozy Dorm Bed ($29/n)": 29,
            "Private Cozy Double ($79/n)": 79,
            "Luxury Penthouse Dorm ($49/n)": 49,
            "Antigravity Theme Suite ($129/n)": 129
        }
        
        rate = room_rates[room_choice]
        room_total = rate * nights
        breakfast_total = (8 * nights * guests) if add_breakfast else 0
        parking_total = (15 * nights) if add_parking else 0
        crawl_total = (20 * guests) if add_crawl else 0
        shuttle_total = 25 if add_shuttle else 0
        
        subtotal = room_total + breakfast_total + parking_total + crawl_total + shuttle_total
        tax = subtotal * 0.10  # 10% VAT
        grand_total = subtotal + tax
        
        st.markdown(f"""
        <div class="booking-card">
            <div class="booking-header">💰 Estimated Quotation</div>
            <div style="font-size: 0.9rem; line-height: 1.6;">
                🏨 Base Room Rate: ${room_total:.2f} ({nights} nights)<br>
                🍳 Breakfast Extras: ${breakfast_total:.2f}<br>
                🚗 Parking: ${parking_total:.2f}<br>
                🍹 Tours/Shuttles: ${crawl_total + shuttle_total:.2f}<br>
                📉 Tax & VAT (10%): ${tax:.2f}
            </div>
            <hr style="margin: 10px 0; border: 0.5px solid rgba(255,255,255,0.1)">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-weight: 700;">Grand Total:</span>
                <span class="booking-price">${grand_total:.2f}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚀 Send Quote to CozyBot", use_container_width=True):
            extras_list = []
            if add_breakfast: extras_list.append("Breakfast Buffet")
            if add_parking: extras_list.append("Parking Space")
            if add_crawl: extras_list.append("Pub Crawl Tour")
            if add_shuttle: extras_list.append("Airport Shuttle")
            
            extras_str = ", ".join(extras_list) if extras_list else "No extras selected"
            
            quote_query = f"I want to book the {room_choice.split(' ($')[0]} for {nights} nights, with {guests} guest(s). Extras selected: {extras_str}. Please verify pricing details, availability, and how to proceed."
            st.session_state.pending_input = quote_query
            st.rerun()

# Main Chat Column
with chat_col:
    # Render suggestion chips / quick prompts
    st.markdown("##### 💡 Quick Suggestions")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        if st.button("🐾 Pet Policies", use_container_width=True):
            st.session_state.pending_input = "What are the rules regarding pets? Can I bring my dog?"
    with c2:
        if st.button("📅 Cancel Booking", use_container_width=True):
            st.session_state.pending_input = "How do I cancel my reservation and is there a fee?"
    with c3:
        if st.button("🍽️ Check Café Menu", use_container_width=True):
            st.session_state.pending_input = "Can I check the café menu and prices?"
    with c4:
        if st.button("🚗 Park Spot Reservation", use_container_width=True):
            st.session_state.pending_input = "Can I reserve a parking space at the hostel?"

    # Display chat messages from history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat Input handler
    user_query = st.chat_input("Ask CozyBot about your stay, pricing, or recommendations...")
    
    # Check if a suggestion chip or simulator button was clicked
    if st.session_state.pending_input:
        user_query = st.session_state.pending_input
        st.session_state.pending_input = None

    if user_query:
        # Display user message in chat
        with st.chat_message("user"):
            st.markdown(user_query)
        st.session_state.messages.append({"role": "user", "content": user_query})
        
        # RAG Retrieval step
        with st.spinner("CozyBot is thinking..."):
            retrieved_contexts = retrieve_relevant_contexts(user_query, model, index, metadata, k=3)
            
            # Format retrieved context for the system prompt
            context_text = ""
            if retrieved_contexts:
                context_text = "\n\n".join([
                    f"Category: {item['category']} | Intent: {item['intent']}\n"
                    f"Instruction: {item['instruction']}\n"
                    f"Response Guideline: {item['response']}"
                    for item in retrieved_contexts
                ])
            else:
                context_text = "No direct match found in database. Respond using general hospitality guidelines."
            
            # Prepare messages list for Llama API
            api_messages = [
                {"role": "system", "content": SYSTEM_PROMPT.format(context_text=context_text)}
            ]
            
            # Append recent chat history (last 5 messages) for memory context
            for msg in st.session_state.messages[-6:-1]:
                api_messages.append({"role": msg["role"], "content": msg["content"]})
                
            api_messages.append({"role": "user", "content": user_query})
            
            try:
                # Call Groq client
                groq_client = get_groq_client()
                completion = groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=api_messages,
                    temperature=0.7,
                    max_tokens=1024
                )
                response_content = completion.choices[0].message.content
            except Exception as e:
                response_content = f"Sorry! I encountered an error communicating with the Groq API: {str(e)}"
                
        # Display assistant response in chat
        with st.chat_message("assistant"):
            st.markdown(response_content)
        st.session_state.messages.append({"role": "assistant", "content": response_content})
        st.rerun()
