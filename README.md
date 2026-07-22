# 🤖 Agentic AI & RAG Chatbot Showcase Suite

Welcome to the **Agentic AI & RAG Chatbot Showcase Suite**! This repository is a curated collection of 12 distinct, premium, and feature-rich AI applications. Designed specifically for educational content, YouTube tutorials, and developers looking to master Agentic workflows, these projects showcase a wide array of state-of-the-art architectures, including Retrieval-Augmented Generation (RAG), Autonomous Coding Agents, ReAct (Reasoning and Action) loops, and custom Tkinter-based GUIs.

Every application features **premium styling**, modern typography, responsive interfaces (using Streamlit or CustomTkinter), and integrations with high-performance LLMs (via Groq, NVIDIA Nemotron, etc.).

---

## 📂 Repository Structure & Overview

The workspace is organized into 12 standalone subfolders:

```bash
Codes/
├── 🌌 1-GroqChatbot/            # AetherChat: Pure Llama-3 Chat Interface
├── 💻 2-AI_AgentCoding/         # AI Coding Agent & File Builder (GUI)
├── 📊 3-AiAgentSearch/          # AI CSV Dataset Search Agent (GUI)
├── ✈️ 4-Travel_RAG/             # AeroQuest: Semantic Travel RAG Chatbot
├── 🛍️ 5-AgenticRAG/             # LuxeCart: Premium E-Commerce Agentic RAG
├── 🍽️ 6-AgenticRAGResto/        # GourmetAgent AI: ReAct Restaurant Concierge
├── 🛡️ 7-CustomerSupport/        # Aegis: Database-Integrated Customer Support
├── 🧠 8-Chatbot_Natural/        # Nemotron Natural Language Agentic RAG
├── 🎫 9-TicketChatbot/          # Helix AI: SQLite-backed Autonomous Ticketing System
├── 🛡️ 10-InsuranceChatbot/      # Insurance Agent Ultra: NVIDIA Nemotron & Plotly Analytics
├── ⚡ 11-TelcoChatbot/          # Antigravity Telecom Agentic RAG & Analytics Hub
└── 🏨 12-HostelAgent/           # CozyBot: SentenceTransformer & FAISS Hostel Guide
```

---

## 🛠️ Project Deep-Dives

### 1. 🌌 AetherChat — Llama 3 70B Chat Interface (`1-GroqChatbot`)
* **Purpose**: A minimalist, high-speed chatbot UI template leveraging Groq Cloud APIs.
* **Tech Stack**: Streamlit, Groq SDK.
* **Key Features**:
  * Premium deep-space purple gradients with glassmorphic accents.
  * Explains model parameters (temperature, max tokens) inside the sidebar.
  * Ideal boilerplate for developers wanting a clean starter template.

### 2. 💻 AI Coding Agent & File Builder (`2-AI_AgentCoding`)
* **Purpose**: A standalone desktop IDE/assistant that writes, tests, and saves Python programs from natural language prompts.
* **Tech Stack**: CustomTkinter (Python Desktop GUI), Groq SDK (Llama models).
* **Key Features**:
  * Direct file writing capabilities with syntax-focused UI.
  * Built-in challenge templates: Fibonacci sequence, Simple Web Scraper, CSV Data Analyzer, CLI To-Do App, and a Tkinter Paint App.
  * Multi-threaded generation to avoid UI freezing.

### 3. 📊 AI CSV Dataset Search Agent (`3-AiAgentSearch`)
* **Purpose**: A desktop application that automatically searches, scrapes, parses, and formats structured datasets into CSV files on-demand.
* **Tech Stack**: CustomTkinter, DuckDuckGo Search, BeautifulSoup4, Pandas.
* **Key Features**:
  * Search preset shortcuts for popular data tasks (Iris, Diabetes, Heart Disease, Titanic, and Credit Card Fraud).
  * Real-time extraction logs displaying agent's step-by-step progress.
  * Quick tabular viewing of completed CSV files.

### 4. ✈️ AeroQuest — Semantic Travel RAG Chatbot (`4-Travel_RAG`)
* **Purpose**: A luxury travel recommendation chatbot that helps users plan itineraries, discover destinations, and query flights or accommodations.
* **Tech Stack**: Streamlit, LangChain, HuggingFace Embeddings (`all-mpnet-base-v2`), FAISS, ChatGroq.
* **Key Features**:
  * Fully vectorized travel catalog based on `travel.csv`.
  * Conversational memory utilizing LangChain's history structures.
  * Interactive UI cards showing details, price points, and ratings.

### 5. 🛍️ LuxeCart — Premium E-Commerce Agentic RAG (`5-AgenticRAG`)
* **Purpose**: A luxurious AI E-commerce assistant capable of searching product catalogs, answering shopping queries, and offering personalized recommendations.
* **Tech Stack**: Streamlit, LangChain, HuggingFace Embeddings, FAISS, ChatGroq, Pandas.
* **Key Features**:
  * Custom Glassmorphic Dark UI using Tailwind-inspired HSL styling.
  * Vector search over `ecommerce.csv` for hybrid keyword and semantic queries.
  * Real-time recommendation cards.

### 6. 🍽️ GourmetAgent AI — ReAct Restaurant Concierge (`6-AgenticRAGResto`)
* **Purpose**: A smart agentic restaurant recommendation system combining local CSV databases with live web search.
* **Tech Stack**: Streamlit, LangGraph (ReAct agent architecture), FAISS, DuckDuckGo Search Tool, ChatGroq.
* **Key Features**:
  * Autonomous tool selection (e.g. deciding whether to query the local `restaurants.csv` index or fetch real-time web menus).
  * Streaming conversational agent replies.
  * Multi-turn reasoning loops.

### 7. 🛡️ Aegis — AI Customer Support Agent (`7-CustomerSupport`)
* **Purpose**: A production-grade support bot linked directly to a customer SQLite database to track orders, update accounts, and retrieve policies.
* **Tech Stack**: Streamlit, SQLite, LangChain, ChatGroq.
* **Key Features**:
  * Read/Write actions over customer records (simulated SQL tools).
  * Agentic routing to handle order status checks, addresses updates, and refund queries.
  * Step-by-step execution trace (thought logs).

### 8. 🧠 Nemotron Natural Agentic RAG (`8-Chatbot_Natural`)
* **Purpose**: An agentic search RAG designed around natural, conversational queries using Llama-3 or Nemotron-based logic.
* **Tech Stack**: Streamlit, Chroma DB, FAISS, Custom Vector Managers.
* **Key Features**:
  * Compare results between local Chroma database and FAISS indexes.
  * Natural language optimization techniques.
  * Visual trace of matched documents.

### 9. 🎫 Helix AI — Autonomous Ticketing Agent (`9-TicketChatbot`)
* **Purpose**: A full-stack ticketing lifecycle workspace. The agent can search, create, update, and resolve service tickets on behalf of the user.
* **Tech Stack**: Streamlit, SQLite (`tickets.db`), FAISS, Plotly.
* **Key Features**:
  * Full CRUD operations on service tickets managed through natural language tools.
  * Dynamic priority classification and status monitoring dashboard.
  * Agentic loop visual logs showing tool invocations, inputs, and execution results.

### 10. 🛡️ Insurance AI Agent Ultra (`10-InsuranceChatbot`)
* **Purpose**: A dashboard and agent suite for processing insurance policies, claim history, and financial trends.
* **Tech Stack**: Streamlit, NVIDIA Nemotron models, Chroma DB, Plotly (Express & Graph Objects).
* **Key Features**:
  * Real-time interactive visualizations showing metrics (e.g., claims distribution, policy costs).
  * Dual-engine approach: Local PDF/CSV RAG fallback to Google/DuckDuckGo web searches.
  * Detailed policy comparison views.

### 11. ⚡ Antigravity Telecom Agentic RAG Hub (`11-TelcoChatbot`)
* **Purpose**: An analytics-heavy support assistant handling telecom subscription inquiries, network complaints, and billing issues.
* **Tech Stack**: Streamlit, LangChain, FAISS, ChatGroq, Plotly.
* **Key Features**:
  * Integration with telecom intents (`bitext-telco-llm-chatbot-training-dataset.csv`).
  * Admin metrics dashboard showing frequent user query intents and customer satisfaction indexes.
  * Responsive layout with sidebar diagnostic tools.

### 12. 🏨 CozyBot — Hostel Companion (`12-HostelAgent`)
* **Purpose**: An intelligent hospitality support chatbot trained on pre-annotated dataset records to answer questions about bookings, amenities, and policies.
* **Tech Stack**: Streamlit, SentenceTransformers, FAISS, Groq API.
* **Key Features**:
  * Direct ingestion of `bitext-hospitality-llm-chatbot-training-dataset.csv`.
  * Advanced index caching for instantaneous matching.
  * Conversational QA refinement.

---

## ⚡ Quick Start

### 🔑 Prerequisites
Make sure you have **Python 3.10+** installed on your system. You will also need a **Groq API Key** (or another LLM provider key if you choose to adapt the models).

### 📥 Installation & Setup

1. **Clone the Repository**:
   ```bash
   git clone <repository_url>
   cd Codes
   ```

2. **Install Dependencies**:
   Each folder contains a `requirements.txt` file (or can be configured with a shared environment). To set up a shared virtual environment:
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```
   
   Install requirements for a specific project (for example, `4-Travel_RAG`):
   ```bash
   pip install -r 4-Travel_RAG/requirements.txt
   ```

3. **Configure Environment Variables**:
   Create a `.env` file or export your API keys globally:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```
   *Note: Some project scripts feature placeholder API keys that you can replace directly or inject via streamlit secrets / environment variables.*

### 🚀 Running the Apps

* **Streamlit Applications**:
  Navigate to the target directory and run Streamlit:
  ```bash
  cd 4-Travel_RAG
  streamlit run app.py
  ```
  
* **CustomTkinter Desktop GUIs**:
  Launch the Python script directly:
  ```bash
  cd 2-AI_AgentCoding
  python app.py
  ```

---

## 🌟 Key Highlights for Learners

* **Vector DB Choices**: Switch between **FAISS** (great for local file-based indices) and **Chroma DB** (excellent for persistent relational vector stores).
* **Agent Frameworks**: Compare raw LangChain calls with advanced state-based graphs like **LangGraph** (see `6-AgenticRAGResto`).
* **Tool Calling**: Learn how agents use function calling to interact with relational databases (SQLite) and search engines (DuckDuckGo).
* **Plotly Visuals**: Combine conversational AI with real-time metric updates to build hybrid analytical workspaces.

Enjoy exploring and hacking these projects! 🚀
