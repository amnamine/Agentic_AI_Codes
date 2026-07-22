import json
from openai import OpenAI
from duckduckgo_search import DDGS

NVIDIA_API_KEY = ""
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"

# Primary model requested by user, with fallback list if needed
PRIMARY_MODEL = "nvidia/nemotron-3-super-120b-a12b"
FALLBACK_MODELS = [
    "nvidia/nemotron-3-ultra-550b-a55b",
    "meta/llama-3.1-70b-instruct",
    "mistralai/mixtral-8x22b-instruct-v0.1"
]

def web_search(query, max_results=3):
    """
    Executes live web search using DuckDuckGo for up-to-date real world info.
    """
    try:
        ddgs = DDGS()
        results = list(ddgs.text(query, max_results=max_results))
        formatted = []
        for r in results:
            formatted.append(f"Title: {r.get('title')}\nSnippet: {r.get('body')}\nURL: {r.get('href')}")
        return "\n\n".join(formatted)
    except Exception as e:
        return f"Web search unavailable: {str(e)}"

def format_system_prompt(rag_context="", web_context=""):
    prompt = """You are InsuranceBot Ultra, an expert AI Insurance Concierge & Risk Analyst powered by NVIDIA Nemotron.

Your objective is to provide precise, polite, highly detailed, and professional guidance on insurance policies, claims, quotes, coverages, auto/health/home/life/pet insurance, and policy management.

Guidelines:
1. Whenever dataset RAG context is provided, prioritize it for procedure steps, portal links, policy terms, and guidelines.
2. If real-time web context is provided, synthesize it seamlessly to give up-to-date industry trends or specific news.
3. Structure your response clearly with Markdown headers, bullet points, numbered steps, and actionable advice.
4. Maintain an authoritative yet empathetic and friendly tone. Always prioritize clarity.
"""
    if rag_context:
        prompt += f"\n\n--- INTERNAL INSURANCE KNOWLEDGE BASE (RAG CONTEXT) ---\n{rag_context}\n--------------------------------------------"
    if web_context:
        prompt += f"\n\n--- REAL-TIME WEB SEARCH RESULTS ---\n{web_context}\n-----------------------------------"
        
    return prompt

def stream_nemotron_agent(messages, rag_context="", web_context="", temperature=0.7, top_p=0.95, reasoning_budget=4096, selected_model=None):
    """
    Streams response and thinking process from NVIDIA API with robust model fallback handling.
    """
    client = OpenAI(base_url=NVIDIA_BASE_URL, api_key=NVIDIA_API_KEY)
    
    system_content = format_system_prompt(rag_context, web_context)
    full_messages = [{"role": "system", "content": system_content}] + messages
    
    models_to_try = [selected_model] if selected_model else [PRIMARY_MODEL] + FALLBACK_MODELS
    # Ensure no duplicates while preserving order
    unique_models = []
    for m in models_to_try:
        if m and m not in unique_models:
            unique_models.append(m)
            
    # Add remaining fallbacks just in case
    for fb in FALLBACK_MODELS:
        if fb not in unique_models:
            unique_models.append(fb)
            
    last_exception = None
    for model_name in unique_models:
        try:
            # Build extra_body based on whether model supports reasoning budget
            extra_body = {}
            if "nemotron" in model_name.lower():
                extra_body = {
                    "chat_template_kwargs": {"enable_thinking": True},
                    "reasoning_budget": int(reasoning_budget)
                }
                
            completion = client.chat.completions.create(
                model=model_name,
                messages=full_messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=2048,
                extra_body=extra_body if extra_body else None,
                stream=True
            )
            
            yield {"active_model": model_name, "reasoning": None, "content": None}
            
            for chunk in completion:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                reasoning = getattr(delta, "reasoning_content", None)
                content = getattr(delta, "content", None)
                
                yield {"active_model": model_name, "reasoning": reasoning, "content": content}
            
            # Successful completion
            return
        except Exception as e:
            last_exception = e
            # Catch API errors (like ResourceExhausted, 429, 503) and proceed to next model in fallback list
            continue
            
    if last_exception:
        raise last_exception
