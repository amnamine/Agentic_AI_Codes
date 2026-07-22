import pandas as pd
import numpy as np
import pickle
import os
import faiss
from sentence_transformers import SentenceTransformer

def build_db():
    csv_path = "bitext-telco-llm-chatbot-training-dataset.csv"
    index_path = "telecom_faiss.index"
    metadata_path = "telecom_metadata.pkl"
    
    print(f"Reading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Dataset loaded. Total rows: {len(df)}")
    
    # Deduplicate to keep representative examples of instructions per intent/category
    print("Deduplicating dataset...")
    df_clean = df.drop_duplicates(subset=["instruction", "intent", "response"])
    print(f"Rows after unique instruction-intent-response deduplication: {len(df_clean)}")
    
    # For speed and efficiency, keep a robust sample of 15,000 diverse rows
    if len(df_clean) > 20000:
        print("Sampling dataset to 15,000 diverse examples for optimal CPU build time...")
        df_clean = df_clean.sample(n=15000, random_state=42).reset_index(drop=True)
    else:
        df_clean = df_clean.reset_index(drop=True)
        
    print(f"Final training dataset size for RAG: {len(df_clean)}")
    
    print("Loading SentenceTransformer model ('all-MiniLM-L6-v2')...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    print("Encoding instructions (this may take 1-2 minutes on CPU)...")
    instructions = df_clean["instruction"].astype(str).tolist()
    embeddings = model.encode(instructions, show_progress_bar=True, convert_to_numpy=True)
    
    # FAISS Index
    dimension = embeddings.shape[1]
    print(f"Creating FAISS Index (Dimension: {dimension})...")
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype("float32"))
    
    # Save index
    print(f"Saving FAISS index to {index_path}...")
    faiss.write_index(index, index_path)
    
    # Save metadata
    print(f"Saving metadata to {metadata_path}...")
    metadata = df_clean[["instruction", "intent", "category", "tags", "response"]].to_dict(orient="records")
    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f)
        
    print("Vector database built successfully!")

if __name__ == "__main__":
    build_db()
