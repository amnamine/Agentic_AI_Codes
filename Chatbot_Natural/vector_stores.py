import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

# Hardcoded API Keys and Local Directories
PINECONE_API_KEY = ""
CHROMA_DIR = "./chroma_db"
FAISS_DIR = "./faiss_index"

class EmbeddingManager:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return self.model.encode(texts, show_progress_bar=False)

    def embed_query(self, text):
        return self.model.encode(text)

class VectorStoreManager:
    def __init__(self, embedding_manager):
        self.embeddings = embedding_manager
        self.chroma_client = None
        self.chroma_collection = None
        self.faiss_index = None
        self.faiss_texts = []
        
    def init_chroma(self):
        """Initialize Chroma DB"""
        import chromadb
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
        self.chroma_collection = self.chroma_client.get_or_create_collection(
            name="natural_questions",
            metadata={"hnsw:space": "cosine"}
        )

    def init_faiss(self, dimension=384):
        """Initialize FAISS Index"""
        import faiss
        # Using Inner Product (Cosine similarity if normalized)
        self.faiss_index = faiss.IndexFlatIP(dimension)
        self.faiss_texts = []

    def index_data(self, df, db_type, pinecone_api_key=None, pinecone_index_name=None, progress_callback=None):
        """Index a dataframe into the specified database type"""
        queries = df['query'].tolist()
        answers = df['answer'].tolist()
        
        # Default to hardcoded key if not provided
        if not pinecone_api_key:
            pinecone_api_key = PINECONE_API_KEY
            
        documents = []
        for q, a in zip(queries, answers):
            documents.append(f"Question: {q}\nAnswer: {a}")
        
        total = len(documents)
        batch_size = 256
        
        if db_type == "Chroma DB":
            self.init_chroma()
            try:
                self.chroma_client.delete_collection("natural_questions")
                self.init_chroma()
            except Exception:
                pass
                
            for i in range(0, total, batch_size):
                batch_docs = documents[i:i+batch_size]
                batch_queries = queries[i:i+batch_size]
                batch_answers = answers[i:i+batch_size]
                batch_embeds = self.embeddings.embed_documents(batch_docs)
                
                ids = [str(idx) for idx in range(i, min(i+batch_size, total))]
                metadatas = [{"query": q, "answer": a} for q, a in zip(batch_queries, batch_answers)]
                
                self.chroma_collection.add(
                    embeddings=batch_embeds.tolist(),
                    documents=batch_docs,
                    metadatas=metadatas,
                    ids=ids
                )
                if progress_callback:
                    progress_callback(min(i + batch_size, total), total)
                    
        elif db_type == "FAISS":
            self.init_faiss()
            all_embeds = []
            
            for i in range(0, total, batch_size):
                batch_docs = documents[i:i+batch_size]
                batch_embeds = self.embeddings.embed_documents(batch_docs)
                faiss_norms = np.linalg.norm(batch_embeds, axis=1, keepdims=True)
                normalized_embeds = batch_embeds / (faiss_norms + 1e-10)
                all_embeds.append(normalized_embeds)
                self.faiss_texts.extend(documents[i:i+batch_size])
                
                if progress_callback:
                    progress_callback(min(i + batch_size, total), total)
            
            if all_embeds:
                stacked = np.vstack(all_embeds).astype('float32')
                self.faiss_index.add(stacked)
                import faiss
                os.makedirs(FAISS_DIR, exist_ok=True)
                faiss.write_index(self.faiss_index, os.path.join(FAISS_DIR, "index.faiss"))
                with open(os.path.join(FAISS_DIR, "texts.txt"), "w", encoding="utf-8") as f:
                    for text in self.faiss_texts:
                        f.write(text.replace("\n", " [NEWLINE] ") + "\n")
                        
        elif db_type == "Pinecone":
            if not pinecone_api_key or not pinecone_index_name:
                raise ValueError("Pinecone API Key and Index Name are required for Pinecone DB.")
            
            from pinecone import Pinecone, ServerlessSpec
            pc = Pinecone(api_key=pinecone_api_key)
            
            if pinecone_index_name not in pc.list_indexes().names():
                pc.create_index(
                    name=pinecone_index_name,
                    dimension=384,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1")
                )
                
            index = pc.Index(pinecone_index_name)
            
            for i in range(0, total, batch_size):
                batch_docs = documents[i:i+batch_size]
                batch_queries = queries[i:i+batch_size]
                batch_answers = answers[i:i+batch_size]
                batch_embeds = self.embeddings.embed_documents(batch_docs)
                
                vectors_to_upsert = []
                for idx, (embed, doc, q, a) in enumerate(zip(batch_embeds, batch_docs, batch_queries, batch_answers)):
                    vectors_to_upsert.append({
                        "id": f"doc_{i+idx}",
                        "values": embed.tolist(),
                        "metadata": {"text": doc, "query": q, "answer": a}
                    })
                
                index.upsert(vectors=vectors_to_upsert)
                if progress_callback:
                    progress_callback(min(i + batch_size, total), total)

    def load_indexes(self, db_type, pinecone_api_key=None, pinecone_index_name=None):
        """Load persistent index from disk or connect to cloud database"""
        if not pinecone_api_key:
            pinecone_api_key = PINECONE_API_KEY
            
        if db_type == "Chroma DB":
            import chromadb
            if os.path.exists(CHROMA_DIR):
                self.chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
                try:
                    self.chroma_collection = self.chroma_client.get_collection(name="natural_questions")
                    return True
                except Exception:
                    return False
            return False
            
        elif db_type == "FAISS":
            import faiss
            index_path = os.path.join(FAISS_DIR, "index.faiss")
            texts_path = os.path.join(FAISS_DIR, "texts.txt")
            if os.path.exists(index_path) and os.path.exists(texts_path):
                self.faiss_index = faiss.read_index(index_path)
                self.faiss_texts = []
                with open(texts_path, "r", encoding="utf-8") as f:
                    for line in f:
                        self.faiss_texts.append(line.strip().replace(" [NEWLINE] ", "\n"))
                return True
            return False
            
        elif db_type == "Pinecone":
            if not pinecone_api_key or not pinecone_index_name:
                return False
            try:
                from pinecone import Pinecone
                pc = Pinecone(api_key=pinecone_api_key)
                if pinecone_index_name in pc.list_indexes().names():
                    return True
            except Exception:
                return False
            return False

    def query(self, query_text, db_type, top_k=3, pinecone_api_key=None, pinecone_index_name=None):
        """Query the selected database and return relevant documents"""
        query_vector = self.embeddings.embed_query(query_text)
        if not pinecone_api_key:
            pinecone_api_key = PINECONE_API_KEY
            
        if db_type == "Chroma DB":
            if not self.chroma_collection:
                self.init_chroma()
            results = self.chroma_collection.query(
                query_embeddings=[query_vector.tolist()],
                n_results=top_k
            )
            documents = results['documents'][0] if results['documents'] else []
            scores = results['distances'][0] if results['distances'] else []
            return [{"text": doc, "score": 1 - min(score, 1.0)} for doc, score in zip(documents, scores)]
            
        elif db_type == "FAISS":
            if self.faiss_index is None:
                if not self.load_indexes("FAISS"):
                    return []
            
            norm = np.linalg.norm(query_vector)
            normalized_query = query_vector / (norm + 1e-10)
            
            import faiss
            query_arr = np.array([normalized_query]).astype('float32')
            scores, indices = self.faiss_index.search(query_arr, top_k)
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.faiss_texts) and idx != -1:
                    results.append({
                        "text": self.faiss_texts[idx],
                        "score": float(score)
                    })
            return results
            
        elif db_type == "Pinecone":
            if not pinecone_api_key or not pinecone_index_name:
                return []
            from pinecone import Pinecone
            pc = Pinecone(api_key=pinecone_api_key)
            index = pc.Index(pinecone_index_name)
            
            res = index.query(
                vector=query_vector.tolist(),
                top_k=top_k,
                include_metadata=True
            )
            
            results = []
            for match in res.matches:
                results.append({
                    "text": match.metadata.get("text", ""),
                    "score": match.score
                })
            return results
        return []
