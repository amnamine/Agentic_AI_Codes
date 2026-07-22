import streamlit as st
import time
from groq import Groq

# Set page configuration with a modern title and icon
st.set_page_config(
    page_title="AetherChat - Llama 3 70B",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
GROQ_API_KEY = ""
MODEL_NAME = "llama-3.3-70b-versatile"

# Inject Custom CSS for Premium UI/UX Design
st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Space+Grotesk:wght@400;500;700&display=swap');

    /* Variables and Theme Overrides */
    :root {
        --primary-gradient: linear-gradient(135deg, #7F00FF 0%, #FF007F 100%);
        --accent-glow: 0 0 20px rgba(127, 0, 255, 0.4);
        --card-bg: rgba(25, 20, 35, 0.65);
        --glass-border: rgba(255, 255, 255, 0.08);
    }

    /* Overall Layout Styling */
    .stApp {
        background: radial-gradient(circle at 50% 50%, #140d21 0%, #08050d 100%) !important;
        font-family: 'Outfit', sans-serif !important;
        color: #f1f1f5 !important;
    }

    /* Titles and Typography */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: -0.5px;
    }

    .main-title {
        background: var(--primary-gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 5px;
        filter: drop-shadow(0 2px 8px rgba(0,0,0,0.5));
    }

    .subtitle {
        color: #a59fb1;
        text-align: center;
        font-size: 1.1rem;
        margin-bottom: 2rem;
        font-weight: 300;
    }

    /* Glassmorphic Cards & Sidebar */
    [data-testid="stSidebar"] {
        background-color: rgba(12, 8, 20, 0.95) !important;
        border-right: 1px solid var(--glass-border) !important;
        backdrop-filter: blur(10px);
    }

    /* Message Bubbles Container */
    .chat-container {
        max-width: 850px;
        margin: 0 auto;
        padding: 10px;
    }

    /* User Message Styling */
    .user-bubble {
        background: linear-gradient(135deg, rgba(127, 0, 255, 0.25) 0%, rgba(255, 0, 127, 0.2) 100%);
        border: 1px solid rgba(255, 0, 127, 0.25);
        color: #ffffff;
        padding: 15px 20px;
        border-radius: 20px 20px 4px 20px;
        margin-bottom: 15px;
        max-width: 80%;
        margin-left: auto;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2), inset 0 0 10px rgba(255, 0, 127, 0.1);
        font-size: 1.05rem;
        animation: slideInRight 0.35s cubic-bezier(0.25, 1, 0.50, 1) forwards;
    }

    /* Assistant Message Styling */
    .assistant-bubble {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid var(--glass-border);
        color: #e5e5ea;
        padding: 15px 20px;
        border-radius: 20px 20px 20px 4px;
        margin-bottom: 15px;
        max-width: 80%;
        margin-right: auto;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.15);
        font-size: 1.05rem;
        animation: slideInLeft 0.35s cubic-bezier(0.25, 1, 0.50, 1) forwards;
        backdrop-filter: blur(5px);
    }

    /* Animations */
    @keyframes slideInRight {
        from {
            opacity: 0;
            transform: translateY(12px) translateX(12px);
        }
        to {
            opacity: 1;
            transform: translateY(0) translateX(0);
        }
    }

    @keyframes slideInLeft {
        from {
            opacity: 0;
            transform: translateY(12px) translateX(-12px);
        }
        to {
            opacity: 1;
            transform: translateY(0) translateX(0);
        }
    }

    /* Input area styling */
    .stChatInputContainer {
        border-radius: 30px !important;
        border: 1px solid var(--glass-border) !important;
        background-color: rgba(20, 15, 30, 0.8) !important;
        backdrop-filter: blur(10px);
        box-shadow: 0 -4px 30px rgba(0, 0, 0, 0.3) !important;
        transition: all 0.3s ease;
    }

    .stChatInputContainer:focus-within {
        border-color: #7F00FF !important;
        box-shadow: 0 0 15px rgba(127, 0, 255, 0.25) !important;
    }

    /* Pulse animation for thinking state */
    .thinking-loader {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 10px 15px;
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid var(--glass-border);
        border-radius: 12px;
        color: #a59fb1;
        font-size: 0.95rem;
        animation: pulse 1.5s infinite ease-in-out;
    }

    @keyframes pulse {
        0%, 100% { opacity: 0.6; }
        50% { opacity: 1; }
    }

    .dot {
        width: 6px;
        height: 6px;
        background-color: #FF007F;
        border-radius: 50%;
        display: inline-block;
        animation: bounce 1.4s infinite ease-in-out both;
    }
    .dot:nth-child(1) { animation-delay: -0.32s; }
    .dot:nth-child(2) { animation-delay: -0.16s; }

    @keyframes bounce {
        0%, 80%, 100% { transform: scale(0); }
        40% { transform: scale(1.0); }
    }
</style>
""", unsafe_allow_html=True)

# Initialize Groq client
@st.cache_resource
def get_groq_client():
    return Groq(api_key=GROQ_API_KEY)

try:
    client = get_groq_client()
except Exception as e:
    st.error(f"Failed to initialize Groq Client: {e}")
    client = None

# Session State Initialization
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar Navigation and Configuration
with st.sidebar:
    st.markdown("<h2 style='text-align: center; color: #ffffff;'>⚙️ Aether Settings</h2>", unsafe_allow_html=True)
    st.markdown("<hr style='border-color: rgba(255,255,255,0.1);'>", unsafe_allow_html=True)

    # Preset System Prompts
    system_presets = {
        "Helpful Assistant": "You are Aether, a highly intelligent, premium AI assistant. Keep responses clear, helpful, structured, and visually clean.",
        "Expert Developer": "You are a master software engineer. Explain concepts with clear code snippets, clean structure, and follow best practices.",
        "Creative Storyteller": "You are a creative writer and storyteller. Use rich metaphors, engaging prose, and immersive descriptions.",
        "Scientific & Analytical": "You are an analytical researcher. Break down answers with scientific accuracy, structured logic, and evidence."
    }
    
    preset_choice = st.selectbox(
        "System Persona",
        options=list(system_presets.keys()),
        index=0
    )
    
    custom_system_prompt = st.text_area(
        "Custom Persona Rules",
        value=system_presets[preset_choice],
        height=100
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # Parameter controls
    temperature = st.slider(
        "Temperature (Creativity)",
        min_value=0.0,
        max_value=1.5,
        value=0.7,
        step=0.1,
        help="Higher values make output more random, lower values make it more deterministic."
    )

    max_tokens = st.slider(
        "Max Output Tokens",
        min_value=256,
        max_value=8192,
        value=4096,
        step=256,
        help="The maximum number of tokens to generate in the reply."
    )

    st.markdown("<hr style='border-color: rgba(255,255,255,0.1);'>", unsafe_allow_html=True)

    # Actions Section
    if st.button("🧹 Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    # Footer
    st.markdown(
        """
        <div style='position: absolute; bottom: 10px; left: 0; right: 0; text-align: center; color: #625c70; font-size: 0.85rem;'>
            Powered by Groq & Llama 3 70B
        </div>
        """,
        unsafe_allow_html=True
    )

# Main Title Area
st.markdown("<div class='main-title'>AETHER CHAT</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Experience state-of-the-art intelligence powered by Llama 3 70B on Groq</div>", unsafe_allow_html=True)

# Render Chat History
st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="user-bubble">{msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="assistant-bubble">{msg["content"]}</div>', unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# User Chat Input
if prompt := st.chat_input("Ask Aether anything..."):
    # Render user prompt immediately
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    st.markdown(f'<div class="user-bubble">{prompt}</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Prepare complete payload for the api call
    api_messages = [{"role": "system", "content": custom_system_prompt}]
    # We append the past chat history
    for msg in st.session_state.messages:
        api_messages.append({"role": msg["role"], "content": msg["content"]})

    # Show loading indicator
    placeholder = st.empty()
    with placeholder.container():
        st.markdown("""
            <div class='chat-container' style='margin-bottom: 15px;'>
                <div class='thinking-loader'>
                    <span>Aether is thinking</span>
                    <span class="dot"></span>
                    <span class="dot"></span>
                    <span class="dot"></span>
                </div>
            </div>
        """, unsafe_allow_html=True)

    # API Call
    if client:
        try:
            chat_completion = client.chat.completions.create(
                messages=api_messages,
                model=MODEL_NAME,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            response_text = chat_completion.choices[0].message.content
        except Exception as e:
            response_text = f"⚠️ An error occurred while generating a response: {str(e)}"
    else:
        response_text = "⚠️ Groq client is not initialized. Please verify API key and dependencies."

    # Clear loader and render the assistant message
    placeholder.empty()
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    st.markdown(f'<div class="assistant-bubble">{response_text}</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Save to history
    st.session_state.messages.append({"role": "assistant", "content": response_text})
    st.rerun()
