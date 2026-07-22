import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

import pandas as pd
import time
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

def main():
    csv_path = r"d:\AMINE2\COURS FAC\LEARNING\YOUTUBER\Codes\4-Travel_RAG\travel.csv"
    index_path = r"d:\AMINE2\COURS FAC\LEARNING\YOUTUBER\Codes\4-Travel_RAG\faiss_index"
    
    print("Loading travel.csv...")
    start_time = time.time()
    df = pd.read_csv(csv_path)
    df['instruction'] = df['instruction'].fillna('')
    df['response'] = df['response'].fillna('')
    print(f"Loaded {len(df):,} rows in {time.time() - start_time:.2f} seconds.")
    
    print("\nPreparing LangChain Documents...")
    documents = []
    for idx, row in df.iterrows():
        doc = Document(
            page_content=row['instruction'],
            metadata={
                "intent": row['intent'],
                "category": row['category'],
                "tags": row['tags'],
                "response": row['response']
            }
        )
        documents.append(doc)
    
    print("\nInitializing Embedding Model (all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    
    print("\nBuilding FAISS Vector Store (this may take 1-2 minutes on CPU)...")
    start_time = time.time()
    vectorstore = FAISS.from_documents(documents, embeddings)
    print(f"Vector Store built in {time.time() - start_time:.2f} seconds.")
    
    print(f"\nSaving vector store to disk at {index_path}...")
    vectorstore.save_local(index_path)
    print("Vector Store saved successfully!")

if __name__ == "__main__":
    main()
