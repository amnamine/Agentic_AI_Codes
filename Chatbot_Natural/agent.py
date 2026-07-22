import time
from openai import OpenAI

# Initialize the OpenAI client for NVIDIA NIM
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=""
)

MODEL_NAME = "nvidia/nemotron-3-super-120b-a12b"

def get_agent_response(query, context_docs, chat_history=[]):
    """
    Formulates a response using agentic RAG and the Nvidia NIM API.
    Yields tuple of (type, content) where type is 'reasoning' or 'content'.
    """
    # Context aggregation
    if context_docs:
        context_str = "\n---\n".join([doc['text'] for doc in context_docs])
    else:
        context_str = "No specific reference documents found."
        
    system_prompt = (
        "You are an Advanced Agentic RAG QA Chatbot designed to answer questions from the Natural Questions dataset.\n"
        "You are grounded on a local knowledge database. Below is the retrieved context from the database.\n"
        "Your task is to analyze the user's question, evaluate the retrieved context for relevance, reason step-by-step, "
        "and provide a highly accurate answer. If the context does not contain the answer, use your general knowledge, "
        "but clearly state that the answer was not found in the retrieved context.\n\n"
        f"Retrieved Context:\n{context_str}\n"
    )
    
    messages = [{"role": "system", "content": system_prompt}]
    
    # Add history
    for msg in chat_history:
        messages.append({"role": msg["role"], "content": msg["content"]})
        
    messages.append({"role": "user", "content": query})
    
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.7,
            top_p=0.9,
            max_tokens=4096,
            extra_body={
                "chat_template_kwargs": {"enable_thinking": True},
                "reasoning_budget": 4096
            },
            stream=True
        )
        
        for chunk in completion:
            if not chunk.choices:
                continue
            
            # Check for reasoning token
            reasoning = getattr(chunk.choices[0].delta, "reasoning_content", None)
            if reasoning:
                yield "reasoning", reasoning
                
            # Check for actual content
            content = chunk.choices[0].delta.content
            if content is not None:
                yield "content", content
                
    except Exception as e:
        yield "error", f"API Error: {str(e)}"
