import os
import pandas as pd
import numpy as np
import chromadb
from chromadb.utils import embedding_functions

CSV_PATH = r"d:\AMINE2\COURS FAC\LEARNING\YOUTUBER\Codes\insuranceCHatbot\insurance.csv"
CHROMA_PATH = r"d:\AMINE2\COURS FAC\LEARNING\YOUTUBER\Codes\insuranceCHatbot\chroma_db"

def init_vector_db(force_rebuild=False, sample_per_intent=150):
    """
    Initializes and builds ChromaDB from insurance.csv dataset.
    Samples representative high-quality Q&A pairs across intents for optimal search performance.
    """
    os.makedirs(CHROMA_PATH, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    
    # Fast lightweight embedding function default in chromadb
    emb_fn = embedding_functions.DefaultEmbeddingFunction()
    
    collection_name = "insurance_knowledge"
    
    existing_collections = [c.name for c in client.list_collections()]
    if collection_name in existing_collections:
        collection = client.get_collection(name=collection_name, embedding_function=emb_fn)
        if not force_rebuild and collection.count() > 0:
            print(f"[VectorDB] Collection '{collection_name}' already exists with {collection.count()} documents.")
            return collection
        else:
            print(f"[VectorDB] Rebuilding collection '{collection_name}'...")
            client.delete_collection(name=collection_name)
            
    collection = client.create_collection(
        name=collection_name,
        embedding_function=emb_fn,
        metadata={"hnsw:space": "cosine"}
    )
    
    print(f"[VectorDB] Reading CSV dataset from {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)
    df = df.dropna(subset=['instruction', 'response']).drop_duplicates(subset=['instruction'])
    
    # Stratified sampling by intent/category for clean representation
    if sample_per_intent and 'intent' in df.columns:
        df_sampled = df.groupby('intent', group_keys=False).apply(
            lambda x: x.sample(min(len(x), sample_per_intent), random_state=42)
        )
    else:
        df_sampled = df.sample(min(len(df), 5000), random_state=42)
        
    print(f"[VectorDB] Indexing {len(df_sampled)} Q&A documents into ChromaDB...")
    
    documents = []
    metadatas = []
    ids = []
    
    for idx, row in df_sampled.iterrows():
        doc_id = f"doc_{idx}"
        instruction = str(row['instruction']).strip()
        response = str(row['response']).strip()
        intent = str(row.get('intent', 'general'))
        category = str(row.get('category', 'GENERAL'))
        tags = str(row.get('tags', ''))
        
        # Combine instruction + response for rich semantic context
        doc_text = f"User Question: {instruction}\nAnswer/Procedure: {response}"
        
        documents.append(doc_text)
        metadatas.append({
            "instruction": instruction,
            "response": response,
            "intent": intent,
            "category": category,
            "tags": tags
        })
        ids.append(doc_id)
        
    # Batch add
    batch_size = 500
    for i in range(0, len(documents), batch_size):
        collection.add(
            documents=documents[i:i+batch_size],
            metadatas=metadatas[i:i+batch_size],
            ids=ids[i:i+batch_size]
        )
        print(f" Indexed batch {i} to {min(i+batch_size, len(documents))}")
        
    print(f"[VectorDB] Ingestion complete. Total items indexed: {collection.count()}")
    return collection

def query_rag(collection, query_text, n_results=4, category_filter=None):
    """
    Queries ChromaDB vector database for relevant insurance contexts.
    """
    where_clause = None
    if category_filter and category_filter != "ALL":
        where_clause = {"category": category_filter}
        
    results = collection.query(
        query_texts=[query_text],
        n_results=n_results,
        where=where_clause
    )
    
    retrieved = []
    if results and 'metadatas' in results and len(results['metadatas']) > 0:
        for meta, doc, dist in zip(results['metadatas'][0], results['documents'][0], results['distances'][0]):
            score = round(1.0 - float(dist), 4) # Cosine similarity score
            retrieved.append({
                "instruction": meta.get("instruction", ""),
                "response": meta.get("response", ""),
                "intent": meta.get("intent", ""),
                "category": meta.get("category", ""),
                "score": score,
                "document": doc
            })
    return retrieved

if __name__ == "__main__":
    coll = init_vector_db(force_rebuild=True)
    print("Testing sample search query...")
    res = query_rag(coll, "How do I file a car accident insurance claim?", n_results=2)
    for r in res:
        print("Score:", r['score'])
        print("Intent:", r['intent'])
        print("Question:", r['instruction'])
        print("Response:", r['response'][:150], "...\n")
