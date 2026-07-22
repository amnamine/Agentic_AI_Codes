from groq import Groq
import json
import re
import streamlit as st
from database import (
    get_order, cancel_order, update_shipping_address,
    get_account, create_account, delete_account,
    update_subscription, request_refund
)
from rag import search_rag, format_rag_results

# Hardcoded API key as requested by the user
GROQ_API_KEY = ""
DEFAULT_MODEL = "llama-3.3-70b-versatile"

def get_groq_client():
    return Groq(api_key=GROQ_API_KEY)

def classify_and_extract(query, history_context=""):
    """
    Uses Llama 3.3 70B to classify the query's category/intent and extract entities in a structured JSON format.
    """
    client = get_groq_client()
    
    prompt = f"""
Analyze the following customer support request and classify it into one of the predefined categories and intents.
Also extract any relevant entities such as order numbers, emails, addresses, names, or refund amounts.

### Categories and Intents:
- ORDER: 'cancel_order', 'change_order', 'place_order', 'track_order'
- SHIPPING: 'change_shipping_address', 'set_up_shipping_address'
- CANCEL: 'check_cancellation_fee'
- INVOICE: 'check_invoice', 'get_invoice'
- PAYMENT: 'check_payment_methods', 'payment_issue'
- REFUND: 'check_refund_policy', 'get_refund', 'track_refund'
- FEEDBACK: 'complaint', 'review'
- CONTACT: 'contact_customer_service', 'contact_human_agent'
- ACCOUNT: 'create_account', 'delete_account', 'edit_account', 'recover_password', 'registration_problems', 'switch_account'
- DELIVERY: 'delivery_options', 'delivery_period'
- SUBSCRIPTION: 'newsletter_subscription'

If the query is a general greeting or does not fit any of the above, use:
- category: "GENERAL"
- intent: "greeting_or_other"

### Output Format (Strict JSON):
Your output must be ONLY a valid JSON block, with no other text, markdown wrapper (except standard json block), or prefix.
{{
  "category": "The classified category (e.g. ORDER)",
  "intent": "The classified intent (e.g. cancel_order)",
  "reasoning": "Brief explanation of why this classification and what the user wants.",
  "entities": {{
    "order_number": "Extract order ID if present (standard format is ORD-XXXXX or digits, e.g. ORD-11502)",
    "email": "Extract email address if present",
    "name": "Extract customer name if present",
    "shipping_address": "Extract complete address if user wants to change/set shipping address",
    "subscribe_action": "true/false if subscription-related",
    "general_entity": "Any other key entity extracted"
  }}
}}

### Context (Recent Conversation History):
{history_context}

### Customer Query:
"{query}"
"""

    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are an expert customer support classifier. You always output strict JSON conforming to the requested schema."},
                {"role": "user", "content": prompt}
            ],
            model=DEFAULT_MODEL,
            temperature=0.0, # Deterministic classification
            response_format={"type": "json_object"}
        )
        
        content = response.choices[0].message.content
        return json.loads(content)
    except Exception as e:
        print(f"Classification error: {e}")
        # Return fallback
        return {
            "category": "GENERAL",
            "intent": "greeting_or_other",
            "reasoning": f"Fallback due to classification error: {str(e)}",
            "entities": {}
        }

def execute_agent_tools(intent, entities, query):
    """
    Executes mock database tools or RAG searches depending on the classified intent and entities.
    Returns:
        - tool_status: Description of actions taken
        - tool_result: Structured result from mock database/RAG
        - rag_context: Raw text context retrieved from the CSV
    """
    tool_status = []
    tool_result = {}
    rag_context = ""
    
    # 1. Fetch guidelines from RAG first for context
    rag_results = search_rag(query, intent=intent, top_k=2)
    rag_context = format_rag_results(rag_results)
    
    # 2. Execute Stateful Database Tools
    if intent == "cancel_order":
        order_num = entities.get("order_number")
        if order_num:
            success, msg = cancel_order(order_num)
            tool_status.append(f"Invoked cancellation tool for Order {order_num}.")
            tool_result["cancel_order"] = {"success": success, "message": msg}
        else:
            tool_status.append("Intent 'cancel_order' detected, but no order number was found in user input.")
            
    elif intent == "change_shipping_address":
        order_num = entities.get("order_number")
        address = entities.get("shipping_address")
        if order_num and address:
            success, msg = update_shipping_address(order_num, address)
            tool_status.append(f"Invoked update shipping address tool for Order {order_num}.")
            tool_result["change_shipping_address"] = {"success": success, "message": msg}
        else:
            tool_status.append("Intent 'change_shipping_address' detected, but missing order number or shipping address.")
            
    elif intent == "track_order":
        order_num = entities.get("order_number")
        if order_num:
            order = get_order(order_num)
            tool_status.append(f"Queried tracking status for Order {order_num}.")
            if order:
                tool_result["track_order"] = {"success": True, "order": order}
            else:
                tool_result["track_order"] = {"success": False, "message": f"Order {order_num} not found."}
        else:
            tool_status.append("Intent 'track_order' detected, but no order number was provided.")
            
    elif intent == "get_refund" or intent == "track_refund":
        order_num = entities.get("order_number")
        if order_num:
            success, msg = request_refund(order_num)
            tool_status.append(f"Processed refund request tool for Order {order_num}.")
            tool_result["refund"] = {"success": success, "message": msg}
        else:
            tool_status.append("Refund intent detected, but no order number was provided.")
            
    elif intent in ["create_account", "registration_problems"]:
        name = entities.get("name")
        email = entities.get("email")
        if name and email:
            success, msg = create_account(name, email)
            tool_status.append(f"Invoked create account tool for {name} ({email}).")
            tool_result["account"] = {"success": success, "message": msg}
        else:
            tool_status.append("Account creation requested, but name or email was not provided.")
            
    elif intent == "delete_account":
        email = entities.get("email")
        if email:
            success, msg = delete_account(email)
            tool_status.append(f"Invoked delete account tool for {email}.")
            tool_result["account"] = {"success": success, "message": msg}
        else:
            tool_status.append("Account deletion requested, but email was not provided.")
            
    elif intent == "newsletter_subscription":
        email = entities.get("email")
        action = entities.get("subscribe_action", "true").lower() != "false"
        if email:
            success, msg = update_subscription(email, action)
            tool_status.append(f"Invoked subscription update tool for {email} (subscribe={action}).")
            tool_result["subscription"] = {"success": success, "message": msg}
        else:
            tool_status.append("Subscription modification requested, but email was not provided.")

    if not tool_status:
        tool_status.append("No database tools triggered. Responding using knowledge base guidance.")
        
    return ", ".join(tool_status), tool_result, rag_context

def synthesize_response(query, classification, tool_status, tool_result, rag_context, history_context=""):
    """
    Synthesizes the final helpful customer support response using Groq.
    """
    client = get_groq_client()
    
    prompt = f"""
You are "Aegis", a premier, highly empathetic, and professional AI Customer Support Agent.
Your goal is to answer the customer's query using standard corporate guidelines (from the RAG context) and the results of any backend tools executed.

### Classification Details:
- Category: {classification.get('category')}
- Intent: {classification.get('intent')}

### Backend Action Log:
- Status: {tool_status}
- Result: {json.dumps(tool_result, indent=2)}

### RAG Reference Guidelines (Standard Responses):
{rag_context}

### IMPORTANT GUIDELINES FOR YOUR RESPONSE:
1. Be polite, warm, and highly professional.
2. If a backend action succeeded (e.g., cancelled order, changed address), clearly state that this action has been completed.
3. Use the RAG guidelines to align your tone and instructions. Ensure you replace templated variables (like {{Order Number}}, {{Customer Support Phone Number}}, etc.) with actual details if available from the backend action results or context.
4. If details (like order number or email) are missing to complete an action, politely ask the user to provide them.
5. If the request was classified as general conversation, respond in a friendly, helpful support manner.

### Context (Recent Conversation History):
{history_context}

### Customer Query:
"{query}"

Please provide your final customer support response:
"""

    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are Aegis, a premier, highly professional customer support AI agent."},
                {"role": "user", "content": prompt}
            ],
            model=DEFAULT_MODEL,
            temperature=0.5
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"I apologize, but I encountered an error communicating with my processing core: {e}. Please try again shortly."

def process_customer_query(query, chat_history=[]):
    """
    Main entry point for processing a customer query.
    Compiles history context, runs the agent flow, and returns both the final message and execution details for the UI.
    """
    # 1. Format history context
    history_lines = []
    for h in chat_history[-6:]:  # Last 3 turns (user + assistant)
        role = "Customer" if h["role"] == "user" else "Agent"
        history_lines.append(f"{role}: {h['content']}")
    history_context = "\n".join(history_lines)
    
    # 2. Intent Classification & Entity Extraction
    classification = classify_and_extract(query, history_context)
    
    # 3. Tool Execution (DB actions + RAG lookup)
    intent = classification.get("intent", "greeting_or_other")
    entities = classification.get("entities", {})
    tool_status, tool_result, rag_context = execute_agent_tools(intent, entities, query)
    
    # 4. Response Synthesis
    final_response = synthesize_response(query, classification, tool_status, tool_result, rag_context, history_context)
    
    # 5. Compile agent trace for UI
    trace = {
        "reasoning": classification.get("reasoning", ""),
        "category": classification.get("category", "UNKNOWN"),
        "intent": intent,
        "entities": entities,
        "tool_status": tool_status,
        "tool_result": tool_result,
        "rag_context": rag_context
    }
    
    return final_response, trace
