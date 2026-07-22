import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# Global cache for data and vectorizers to avoid reloading
_df = None
_vectorizers = {}  # Cache of vectorizers per intent
_tfidf_matrices = {}  # Cache of tf-idf matrices per intent
_intent_dfs = {}  # Cache of filtered dataframes per intent

def load_data(csv_path="customersupport.csv"):
    global _df
    if _df is None:
        try:
            _df = pd.read_csv(csv_path)
            # Fill NaNs
            _df["instruction"] = _df["instruction"].fillna("")
            _df["response"] = _df["response"].fillna("")
            _df["intent"] = _df["intent"].fillna("")
            _df["category"] = _df["category"].fillna("")
        except Exception as e:
            print(f"Error loading CSV data: {e}")
            # Fallback empty dataframe
            _df = pd.DataFrame(columns=["flags", "instruction", "category", "intent", "response"])
    return _df

def get_intent_data(intent, csv_path="customersupport.csv"):
    """
    Returns filtered dataframe, tfidf vectorizer, and tfidf matrix for a specific intent.
    Caches the result to ensure subsequent queries are instantaneous.
    """
    global _vectorizers, _tfidf_matrices, _intent_dfs
    
    df = load_data(csv_path)
    intent_clean = str(intent).strip().lower()
    
    # Check if cached
    if intent_clean in _intent_dfs:
        return _intent_dfs[intent_clean], _vectorizers[intent_clean], _tfidf_matrices[intent_clean]
    
    # Filter by intent
    intent_df = df[df["intent"].str.lower() == intent_clean].copy()
    
    # If intent is not found, fallback to category search or return full dataset sample
    if intent_df.empty:
        # Check if it matches a category
        intent_df = df[df["category"].str.lower() == intent_clean].copy()
        if intent_df.empty:
            # Absolute fallback: sample of the whole dataset
            intent_df = df.sample(n=min(1000, len(df)), random_state=42).copy()
            
    # Reset index
    intent_df = intent_df.reset_index(drop=True)
    
    # Create TF-IDF index
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(intent_df["instruction"])
    
    # Cache
    _vectorizers[intent_clean] = vectorizer
    _tfidf_matrices[intent_clean] = tfidf_matrix
    _intent_dfs[intent_clean] = intent_df
    
    return intent_df, vectorizer, tfidf_matrix

def search_rag(query, intent=None, top_k=3, csv_path="customersupport.csv"):
    """
    Queries the RAG system. Returns a list of dicts with matching instructions and responses.
    """
    try:
        if not intent:
            # If no intent is specified, perform a global search (using a subset or full TF-IDF if small)
            intent = "global_fallback"
            
        intent_df, vectorizer, tfidf_matrix = get_intent_data(intent, csv_path)
        
        if intent_df.empty or tfidf_matrix.shape[0] == 0:
            return []
            
        # Vectorize query
        query_vec = vectorizer.transform([query])
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
        
        # Get top K indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            # Only include reasonable matches or all if top_k is small
            row = intent_df.iloc[idx]
            results.append({
                "instruction": row["instruction"],
                "response": row["response"],
                "category": row["category"],
                "intent": row["intent"],
                "score": score
            })
        return results
    except Exception as e:
        print(f"RAG search error: {e}")
        return []

def format_rag_results(rag_results):
    if not rag_results:
        return "No relevant reference guidelines found in the dataset."
        
    formatted = "### Retreived Reference Responses & Instructions:\n"
    for i, res in enumerate(rag_results, 1):
        formatted += f"\n**Reference {i}** (Confidence: {res['score']:.2f} | Category: {res['category']} | Intent: {res['intent']})\n"
        formatted += f"- *Customer Ask pattern:* {res['instruction']}\n"
        formatted += f"- *Standard Response pattern:* {res['response']}\n"
    return formatted
