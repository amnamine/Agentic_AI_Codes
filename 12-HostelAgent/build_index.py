import pandas as pd
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import os

def main():
    csv_path = "bitext-hospitality-llm-chatbot-training-dataset.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    print("Loading dataset...")
    df = pd.read_csv(csv_path)
    
    # Drop duplicate instructions to keep indexing clean and semantic search unique
    print("Removing duplicates...")
    df_clean = df.drop_duplicates(subset=['instruction']).copy()
    print(f"Cleaned dataset from {len(df)} to {len(df_clean)} rows.")

    instructions = df_clean['instruction'].tolist()
    
    print("Loading SentenceTransformer model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print("Computing embeddings...")
    embeddings = model.encode(instructions, show_progress_bar=True, batch_size=128)
    embeddings = np.array(embeddings).astype('float32')
    
    print("Creating FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    print("Saving FAISS index...")
    faiss.write_index(index, "vector_store.faiss")
    
    print("Saving metadata...")
    metadata = df_clean[['instruction', 'intent', 'category', 'response']].to_dict(orient='records')
    with open("vector_store_metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)
        
    print("Done! Index and metadata saved successfully.")

if __name__ == "__main__":
    main()
