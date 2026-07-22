import os
import pickle
import faiss
import numpy as np
from groq import Groq
from sentence_transformers import SentenceTransformer

# Hardcoded Groq API key as requested by the user
GROQ_API_KEY = ""
client = Groq(api_key=GROQ_API_KEY)

# Lazy loading of FAISS and embedding model
_embedding_model = None
_faiss_index = None
_metadata = None

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedding_model

def load_vector_db():
    global _faiss_index, _metadata
    if _faiss_index is None:
        index_path = "telecom_faiss.index"
        metadata_path = "telecom_metadata.pkl"
        if os.path.exists(index_path) and os.path.exists(metadata_path):
            _faiss_index = faiss.read_index(index_path)
            with open(metadata_path, "rb") as f:
                _metadata = pickle.load(f)
        else:
            raise FileNotFoundError("Vector DB index or metadata files not found. Run build_vector_db.py first.")
    return _faiss_index, _metadata

def retrieve_telecom_policy(query: str, top_k: int = 3):
    """
    Search the vector DB for policies, instructions, and standard responses.
    """
    try:
        index, metadata = load_vector_db()
        model = get_embedding_model()
        
        query_vector = model.encode([query], convert_to_numpy=True).astype("float32")
        distances, indices = index.search(query_vector, top_k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(metadata):
                item = metadata[idx]
                results.append({
                    "instruction": item["instruction"],
                    "intent": item["intent"],
                    "category": item["category"],
                    "tags": item["tags"],
                    "response": item["response"],
                    "confidence": float(1.0 / (1.0 + distances[0][i])) # standard similarity score mapping
                })
        return results
    except Exception as e:
        return [{"error": f"Failed to retrieve context: {str(e)}"}]

def billing_simulator(account_id: str, dispute_amount: float, category: str = "Disputed Charge"):
    """
    Simulates a billing audit. Calculates potential credits, refunds, and adjustments.
    """
    # Simple logic simulating advanced billing calculations
    import random
    base_tax_rate = 0.12
    eligible_refund = round(dispute_amount * random.choice([0.5, 0.8, 1.0]), 2)
    audit_id = f"AUD-{random.randint(100000, 999999)}"
    
    return {
        "status": "APPROVED_FOR_CREDIT",
        "audit_id": audit_id,
        "dispute_amount": dispute_amount,
        "eligible_refund": eligible_refund,
        "tax_adjustment": round(eligible_refund * base_tax_rate, 2),
        "notes": f"System audited account {account_id}. The billing error is matched under category: {category}."
    }

def create_support_ticket(user_id: str, issue_description: str, urgency: str = "medium"):
    """
    Create a mock tracking ticket for system engineers to follow up.
    """
    import random
    from datetime import datetime
    ticket_id = f"TEL-{random.randint(10000, 99999)}"
    return {
        "ticket_id": ticket_id,
        "user_id": user_id,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "priority": urgency.upper(),
        "status": "OPEN",
        "assigned_to": "Tier-2 Technical Support",
        "estimated_resolution": "24 Hours"
    }
# Agent reasoning loop using Llama 3.3 70b via Groq
def run_agentic_rag(user_query: str, chat_history: list = None, tone: str = "Empathetic Support"):
    """
    Advanced agent reasoning loop.
    Decides whether to search policies, run the billing simulator, file a support ticket, or response directly.
    """
    if chat_history is None:
        chat_history = []
        
    system_prompt = f"""You are the Antigravity Advanced Telecom AI Assistant, an agentic AI operating over a RAG database.
Please adopt a {tone} tone in your responses.
You have access to the following tools:

1. Retrieve Telecom Policy & FAQ:
   Use this to search the internal knowledge base for template responses, standard procedures, customer answers, categories, and instruction guidelines.
   Always try to call this tool if the user is asking about network issues, billing, subscriptions, phone bill disputes, internet connections, or cancellation.
   Format: RETRIEVE: <search query>

2. Bill Dispute Simulator:
   Use this to simulate a billing audit when a user disputes a charge or asks about bill refund eligibility.
   Format: BILL_AUDIT: account_id=<id>, amount=<amount>, reason=<reason>

3. Create Support Ticket:
   Use this if the customer issue cannot be resolved instantly (e.g., hardware fault, technician visit needed, contract cancel, or escalation).
   Format: CREATE_TICKET: user_id=<id>, issue=<description>, priority=<low/medium/high>

Analyze the user query. Choose the appropriate tool(s) if needed, then output your reasoning step-by-step using thought logs, and compile your final answer.
Your output MUST contain your thinking steps enclosed in <thought>...</thought> tags, followed by your final helpful response.
If you call a tool, state the tool command inside your thought block.
"""
    
    messages = [{"role": "system", "content": system_prompt}]
    for msg in chat_history[-6:]:  # include recent context
        messages.append(msg)
    messages.append({"role": "user", "content": user_query})
    
    # 1. Ask Groq for the action step
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.2
        )
        response_content = completion.choices[0].message.content
    except Exception as e:
        return f"Groq API Error: {str(e)}", ["Error accessing LLM API."]
    
    # Process tools based on LLM response/thinking
    thought_logs = []
    final_text = response_content
    
    # Check if LLM requested a tool call in its text
    if "RETRIEVE:" in response_content:
        # Extract search query
        import re
        match = re.search(r"RETRIEVE:\s*(.*)", response_content)
        if match:
            search_query = match.group(1).split("\n")[0].strip()
            thought_logs.append(f"🔍 Routing to RAG Knowledge Base. Query: '{search_query}'")
            retrieved = retrieve_telecom_policy(search_query)
            
            # Format context
            context_str = "\n".join([
                f"- Intent: {r['intent']}\n  Category: {r['category']}\n  Tags: {r['tags']}\n  Standard Response: {r['response']}"
                for r in retrieved
            ])
            
            # Request LLM to synthesize final answer with the context
            synthesis_messages = messages + [
                {"role": "assistant", "content": response_content},
                {"role": "user", "content": f"Here is the context retrieved from the database:\n{context_str}\n\nSynthesize the final friendly response using this information. Fill templates like {{WEBSITE_URL}}, {{INVOICE_SECTION}}, {{DISPUTE_INVOICE_OPTION}} with realistic names."}
            ]
            try:
                synth_comp = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=synthesis_messages,
                    temperature=0.2
                )
                final_text = synth_comp.choices[0].message.content
            except Exception as e:
                final_text = response_content + f"\n\n(RAG Context retrieved, but failed to synthesize: {str(e)})"
                
    elif "BILL_AUDIT:" in response_content:
        import re
        thought_logs.append("💳 Initiating automated billing audit simulator...")
        # Parse params
        acc_match = re.search(r"account_id=(\w+)", response_content)
        amt_match = re.search(r"amount=([\d.]+)", response_content)
        
        acc_id = acc_match.group(1) if acc_match else "ACC-9999"
        amount = float(amt_match.group(1)) if amt_match else 50.0
        
        audit_res = billing_simulator(acc_id, amount)
        thought_logs.append(f"✅ Audit Complete. Audit ID: {audit_res['audit_id']}. Approved refund: ${audit_res['eligible_refund']}.")
        
        # Synthesize final answer with audit results
        synthesis_messages = messages + [
            {"role": "assistant", "content": response_content},
            {"role": "user", "content": f"The billing audit system returned: {str(audit_res)}. Present this beautifully to the user."}
        ]
        try:
            synth_comp = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=synthesis_messages,
                temperature=0.2
            )
            final_text = synth_comp.choices[0].message.content
        except Exception as e:
            final_text = f"Audit Success. Details: {str(audit_res)}"
            
    elif "CREATE_TICKET:" in response_content:
        import re
        thought_logs.append("🎫 Creating a support ticket for technical escalation...")
        usr_match = re.search(r"user_id=(\w+)", response_content)
        issue_match = re.search(r"issue=(.*)", response_content)
        
        usr_id = usr_match.group(1) if usr_match else "USR-UNKNOWN"
        issue = issue_match.group(1).split(",")[0].strip() if issue_match else "General Telecom issue"
        
        ticket_res = create_support_ticket(usr_id, issue)
        thought_logs.append(f"🎟️ Ticket registered. Ticket ID: {ticket_res['ticket_id']}. Assigned: {ticket_res['assigned_to']}.")
        
        synthesis_messages = messages + [
            {"role": "assistant", "content": response_content},
            {"role": "user", "content": f"The ticket system returned: {str(ticket_res)}. Present this ticket confirmation beautifully to the user."}
        ]
        try:
            synth_comp = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=synthesis_messages,
                temperature=0.2
            )
            final_text = synth_comp.choices[0].message.content
        except Exception as e:
            final_text = f"Ticket Created: {str(ticket_res)}"
            
    else:
        thought_logs.append("💬 Direct response routed (No tool required).")
        
    return final_text, thought_logs
