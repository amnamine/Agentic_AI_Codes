import os
import json
from google import genai
from google.genai import types
from google.genai.errors import APIError
from db_manager import (
    VectorSearcher,
    create_ticket,
    update_ticket,
    get_ticket,
    list_all_tickets
)

API_KEY = ""
MODEL_NAME = "gemini-2.5-flash"

# Helper check for API initialization
def get_genai_client():
    if not API_KEY or API_KEY == "YOUR_API_KEY_HERE":
        raise ValueError("Invalid Gemini API Key. Please provide a valid key.")
    return genai.Client(api_key=API_KEY)

# Define tools
def search_faq(query: str) -> str:
    """Search the FAISS vector database containing ticketing FAQs and knowledge base.
    Use this to look up standard responses, cancellation instructions, refund procedures, etc.
    """
    try:
        searcher = VectorSearcher()
        results = searcher.search(query, top_k=2)
        if not results:
            return "No matching FAQ articles found."
        
        formatted_results = []
        for i, res in enumerate(results):
            formatted_results.append(
                f"Result {i+1}:\n"
                f"- Category: {res.get('category')}\n"
                f"- Intent: {res.get('intent')}\n"
                f"- Match Query: {res.get('instruction')}\n"
                f"- Suggested Response: {res.get('response')}\n"
            )
        return "\n---\n".join(formatted_results)
    except Exception as e:
        return f"Error searching vector database: {str(e)}"

def list_tickets(status: str = None, priority: str = None) -> str:
    """Retrieve the list of customer support tickets currently in the database.
    Optionally filter by status (Open, In Progress, Resolved) or priority (Low, Medium, High).
    """
    try:
        tickets = list_all_tickets(status=status, priority=priority)
        if not tickets:
            return "No tickets found matching the criteria."
        return json.dumps(tickets, indent=2)
    except Exception as e:
        return f"Error listing tickets: {str(e)}"

def create_support_ticket(customer_name: str, customer_email: str, subject: str, description: str, priority: str = "Medium", category: str = "General") -> str:
    """Create a new support ticket in the database.
    Use this if the customer's issue cannot be resolved using the knowledge base FAQ,
    or if they explicitly ask to open a ticket.
    """
    try:
        ticket_id = create_ticket(
            name=customer_name,
            email=customer_email,
            subject=subject,
            description=description,
            priority=priority,
            category=category
        )
        return f"Ticket successfully created. Ticket ID: {ticket_id}"
    except Exception as e:
        return f"Error creating ticket: {str(e)}"

def get_ticket_details(ticket_id: str) -> str:
    """Retrieve full details of a ticket, including description, category, creation date, status, priority, and resolution notes, using the Ticket ID (e.g. TKT-1001)."""
    try:
        t = get_ticket(ticket_id)
        if not t:
            return f"Ticket with ID {ticket_id} not found."
        return json.dumps(t, indent=2)
    except Exception as e:
        return f"Error retrieving ticket details: {str(e)}"

def update_ticket_status(ticket_id: str, status: str = None, priority: str = None, resolution_notes: str = None) -> str:
    """Update ticket properties. Can modify status ('Open', 'In Progress', 'Resolved'), priority ('Low', 'Medium', 'High'), or add resolution notes when resolving the ticket."""
    try:
        success = update_ticket(ticket_id, status=status, priority=priority, resolution_notes=resolution_notes)
        if success:
            return f"Ticket {ticket_id} updated successfully."
        else:
            return f"Ticket {ticket_id} could not be updated (check if ID exists)."
    except Exception as e:
        return f"Error updating ticket: {str(e)}"

# Mapping dictionary for manual function routing
TOOLS_MAP = {
    "search_faq": search_faq,
    "list_tickets": list_tickets,
    "create_support_ticket": create_support_ticket,
    "get_ticket_details": get_ticket_details,
    "update_ticket_status": update_ticket_status
}

SYSTEM_INSTRUCTION = """You are an advanced, autonomous AI Ticketing Agent.
Your role is to assist users with support tickets, answer questions by searching the FAISS knowledge base, and perform ticket management using tools.

Guidelines:
1. When asked a general support question (e.g. "how do I cancel", "refund policy"), search the FAQ vector database first using search_faq.
2. If the user wants to create, view, list, or update tickets, use the corresponding database tool.
3. Keep answers friendly, professional, and clear.
4. When you call a tool, state clearly in your final response what actions were performed so the user has full transparency.
"""

def run_agent_loop(user_message: str, chat_history: list = None, custom_api_key: str = None):
    """Runs a conversational turn with agent tool executing loop."""
    key_to_use = custom_api_key if custom_api_key else API_KEY
    if not key_to_use or key_to_use == "YOUR_API_KEY_HERE":
        yield "Error: Please configure a valid Google Gemini API Key in the sidebar.", []
        return
        
    try:
        client = genai.Client(api_key=key_to_use)
    except Exception as e:
        yield f"Error initializing client: {str(e)}", []
        return

    if chat_history is None:
        chat_history = []
    
    # Format message history for Google GenAI client
    messages = []
    for msg in chat_history:
        messages.append(
            types.Content(
                role=msg["role"],
                parts=[types.Part.from_text(text=msg["content"])]
            )
        )
    # Add new user message
    messages.append(
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=user_message)]
        )
    )
    
    # Configure tools
    tools = [
        types.Tool(
            function_declarations=[
                types.FunctionDeclaration(
                    name="search_faq",
                    description="Search the FAISS vector database containing ticketing FAQs and knowledge base.",
                    parameters=types.Schema(
                        type="OBJECT",
                        properties={
                            "query": types.Schema(type="STRING", description="The search query to match FAQ entries.")
                        },
                        required=["query"]
                    )
                ),
                types.FunctionDeclaration(
                    name="list_tickets",
                    description="Retrieve a list of support tickets from the database, optionally filtered.",
                    parameters=types.Schema(
                        type="OBJECT",
                        properties={
                            "status": types.Schema(type="STRING", description="Filter by status (Open, In Progress, Resolved)."),
                            "priority": types.Schema(type="STRING", description="Filter by priority (Low, Medium, High).")
                        }
                    )
                ),
                types.FunctionDeclaration(
                    name="create_support_ticket",
                    description="Create a new ticket in the database if the issue can't be resolved with FAQ.",
                    parameters=types.Schema(
                        type="OBJECT",
                        properties={
                            "customer_name": types.Schema(type="STRING"),
                            "customer_email": types.Schema(type="STRING"),
                            "subject": types.Schema(type="STRING"),
                            "description": types.Schema(type="STRING"),
                            "priority": types.Schema(type="STRING", description="Low, Medium, High"),
                            "category": types.Schema(type="STRING")
                        },
                        required=["customer_name", "customer_email", "subject", "description"]
                    )
                ),
                types.FunctionDeclaration(
                    name="get_ticket_details",
                    description="Get all details for a ticket by ID.",
                    parameters=types.Schema(
                        type="OBJECT",
                        properties={
                            "ticket_id": types.Schema(type="STRING", description="ID of ticket, e.g. TKT-1001")
                        },
                        required=["ticket_id"]
                    )
                ),
                types.FunctionDeclaration(
                    name="update_ticket_status",
                    description="Update ticket status, priority, or resolution notes.",
                    parameters=types.Schema(
                        type="OBJECT",
                        properties={
                            "ticket_id": types.Schema(type="STRING"),
                            "status": types.Schema(type="STRING", description="Open, In Progress, Resolved"),
                            "priority": types.Schema(type="STRING", description="Low, Medium, High"),
                            "resolution_notes": types.Schema(type="STRING")
                        },
                        required=["ticket_id"]
                    )
                )
            ]
        )
    ]

    FALLBACK_MODELS = [
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        "gemini-1.5-flash",
        "gemini-1.5-flash-8b"
    ]

    # Running execution loop (maximum 5 turns of tool calls to prevent infinite loops)
    for turn in range(5):
        response = None
        last_err = ""
        
        # Sequentially attempt models in the fallback chain
        for model_name in FALLBACK_MODELS:
            # 1. Attempt with thinking budget = 0 (disable thinking)
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=messages,
                    config=types.GenerateContentConfig(
                        tools=tools,
                        system_instruction=SYSTEM_INSTRUCTION,
                        thinking_config=types.ThinkingConfig(thinking_budget=0)
                    )
                )
                break
            except Exception as e:
                err_str = str(e)
                # 2. Attempt without thinking config in case of validation/schema error on older models
                if "thinking_config" in err_str or "thinking" in err_str or "Thinking" in err_str:
                    try:
                        response = client.models.generate_content(
                            model=model_name,
                            contents=messages,
                            config=types.GenerateContentConfig(
                                tools=tools,
                                system_instruction=SYSTEM_INSTRUCTION
                            )
                        )
                        break
                    except Exception as inner_e:
                        last_err = f"Model {model_name} error: {str(inner_e)}"
                        continue
                else:
                    last_err = f"Model {model_name} error: {str(e)}"
                    continue
        else:
            yield f"Error: All Gemini fallback models failed. Last reported error: {last_err}", []
            return
        
        # Check if model wants to call a tool
        function_calls = response.function_calls
        if not function_calls:
            # No tool call, yield final response text
            final_text = response.text or ""
            yield final_text, []
            return

        # We have function calls, process them
        tool_results = []
        messages.append(response.candidates[0].content) # Add assistant content with function calls
        
        for call in function_calls:
            func_name = call.name
            func_args = call.args
            
            tool_msg = f"Executing Tool: `{func_name}` with arguments: {json.dumps(func_args)}"
            tool_results.append(tool_msg)
            
            if func_name in TOOLS_MAP:
                result = TOOLS_MAP[func_name](**func_args)
            else:
                result = f"Error: Tool '{func_name}' is not recognized."
                
            # Create the response part to send back to model
            function_response = types.Part(
                function_response=types.FunctionResponse(
                    name=func_name,
                    response={"result": result}
                )
            )
            
            # Append the response from function execution
            messages.append(
                types.Content(
                    role="tool",
                    parts=[function_response]
                )
            )
        
        # Return tool execution details to UI to show agent step-by-step
        yield None, tool_results
        
    yield "Error: Agent exceeded maximum execution turns.", []
