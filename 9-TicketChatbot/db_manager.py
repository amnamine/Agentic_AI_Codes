import os
import sqlite3
import pandas as pd
import numpy as np
import faiss
import pickle
from datetime import datetime
from sentence_transformers import SentenceTransformer

# Paths
DB_PATH = "tickets.db"
FAISS_INDEX_PATH = "faiss_index.bin"
METADATA_PATH = "faiss_metadata.pkl"
CSV_PATH = "ticketing.csv"

def init_sqlite_db():
    """Initializes the SQLite database with mock tickets."""
    print("Initializing SQLite Database...")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create tickets table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tickets (
            ticket_id TEXT PRIMARY KEY,
            customer_name TEXT NOT NULL,
            customer_email TEXT NOT NULL,
            subject TEXT NOT NULL,
            description TEXT NOT NULL,
            status TEXT NOT NULL,
            priority TEXT NOT NULL,
            category TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            resolution_notes TEXT
        )
    """)
    
    # Check if table already has tickets, if not insert mock data
    cursor.execute("SELECT COUNT(*) FROM tickets")
    if cursor.fetchone()[0] == 0:
        mock_tickets = [
            (
                "TKT-1001",
                "Alice Johnson",
                "alice.j@example.com",
                "Refund Request for Concert",
                "I bought a ticket for the concert next Saturday, but I can no longer make it. Can I get a full refund?",
                "Open",
                "Medium",
                "Refunds",
                "2026-07-19 10:15:00",
                "2026-07-19 10:15:00",
                None
            ),
            (
                "TKT-1002",
                "Bob Smith",
                "bob.smith@example.com",
                "Unable to download digital ticket PDF",
                "Whenever I click on the download link in my confirmation email, I get a 404 page not found error.",
                "In Progress",
                "High",
                "Technical Support",
                "2026-07-19 14:30:00",
                "2026-07-20 09:00:00",
                "Assigned to engineering team to check link generation logic."
            ),
            (
                "TKT-1003",
                "Charlie Brown",
                "charlie.b@example.com",
                "Change attendee name on ticket",
                "I purchased tickets for my family but misspelt my son's name. Can you please correct it from Charli to Charlie?",
                "Resolved",
                "Low",
                "Ticket Changes",
                "2026-07-18 11:00:00",
                "2026-07-18 16:45:00",
                "Updated attendee name in the ticketing system and re-sent the confirmation email."
            ),
            (
                "TKT-1004",
                "Diana Prince",
                "diana.prince@example.com",
                "VIP Lounge access inquiry",
                "I bought a VIP ticket for the upcoming sports match and wanted to know if VIP Lounge access is included.",
                "Open",
                "Low",
                "Inquiries",
                "2026-07-20 11:20:00",
                "2026-07-20 11:20:00",
                None
            ),
            (
                "TKT-1005",
                "Evan Wright",
                "evan.w@example.com",
                "Double charge on credit card",
                "I noticed two identical charges of $120.00 on my credit card statement for transaction ID 874628.",
                "Open",
                "High",
                "Billing",
                "2026-07-20 12:00:00",
                "2026-07-20 12:00:00",
                None
            )
        ]
        cursor.executemany("""
            INSERT INTO tickets (ticket_id, customer_name, customer_email, subject, description, status, priority, category, created_at, updated_at, resolution_notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, mock_tickets)
        conn.commit()
        print("Mock tickets inserted successfully.")
    
    conn.close()

def init_faiss_rag():
    """Initializes the FAISS vector database by reading and indexing ticketing.csv."""
    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(METADATA_PATH):
        print("FAISS index and metadata already exist. Skipping index generation.")
        return

    print("Building FAISS Vector Database from ticketing.csv...")
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Source file {CSV_PATH} not found.")

    # Load CSV
    df = pd.read_csv(CSV_PATH)
    
    # Drop duplicates to keep unique responses/questions to keep DB clean and fast
    df = df.drop_duplicates(subset=["instruction", "response"]).reset_index(drop=True)
    
    # Sample if too large to ensure quick embeddings and low memory usage
    sample_size = min(3000, len(df))
    df_sample = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
    print(f"Sampled {len(df_sample)} rows for vector database.")

    # Load SentenceTransformer model
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Create embeddings
    instructions = df_sample["instruction"].tolist()
    print("Generating embeddings...")
    embeddings = model.encode(instructions, show_progress_bar=True, batch_size=64)
    
    # Initialize FAISS Index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype("float32"))
    
    # Save Index
    faiss.write_index(index, FAISS_INDEX_PATH)
    
    # Save metadata (instruction, intent, category, response)
    metadata = df_sample[["instruction", "intent", "category", "response"]].to_dict(orient="records")
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(metadata, f)
        
    print("FAISS Vector Database built and saved successfully.")

class VectorSearcher:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = faiss.read_index(FAISS_INDEX_PATH)
        with open(METADATA_PATH, "rb") as f:
            self.metadata = pickle.load(f)
            
    def search(self, query: str, top_k: int = 3):
        query_vector = self.model.encode([query]).astype("float32")
        distances, indices = self.index.search(query_vector, top_k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.metadata):
                res = self.metadata[idx].copy()
                res["distance"] = float(distances[0][i])
                results.append(res)
        return results

# Helper functions for ticket operations
def get_db_connection():
    return sqlite3.connect(DB_PATH)

def create_ticket(name, email, subject, description, priority="Medium", category="General"):
    conn = get_db_connection()
    cursor = conn.cursor()
    # Generate new ID
    cursor.execute("SELECT MAX(CAST(SUBSTR(ticket_id, 5) AS INTEGER)) FROM tickets")
    max_id = cursor.fetchone()[0]
    next_id = 1001 if max_id is None else max_id + 1
    ticket_id = f"TKT-{next_id}"
    
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("""
        INSERT INTO tickets (ticket_id, customer_name, customer_email, subject, description, status, priority, category, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (ticket_id, name, email, subject, description, "Open", priority, category, now, now))
    conn.commit()
    conn.close()
    return ticket_id

def update_ticket(ticket_id, status=None, priority=None, resolution_notes=None):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    updates = []
    params = []
    if status:
        updates.append("status = ?")
        params.append(status)
    if priority:
        updates.append("priority = ?")
        params.append(priority)
    if resolution_notes:
        updates.append("resolution_notes = ?")
        params.append(resolution_notes)
        
    if not updates:
        conn.close()
        return False
        
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    updates.append("updated_at = ?")
    params.append(now)
    
    params.append(ticket_id)
    query = f"UPDATE tickets SET {', '.join(updates)} WHERE ticket_id = ?"
    cursor.execute(query, params)
    conn.commit()
    rows_affected = cursor.rowcount
    conn.close()
    return rows_affected > 0

def get_ticket(ticket_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM tickets WHERE ticket_id = ?", (ticket_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        columns = [col[0] for col in cursor.description]
        return dict(zip(columns, row))
    return None

def list_all_tickets(status=None, priority=None):
    conn = get_db_connection()
    cursor = conn.cursor()
    query = "SELECT * FROM tickets"
    params = []
    filters = []
    if status:
        filters.append("status = ?")
        params.append(status)
    if priority:
        filters.append("priority = ?")
        params.append(priority)
        
    if filters:
        query += " WHERE " + " AND ".join(filters)
        
    query += " ORDER BY created_at DESC"
    cursor.execute(query, params)
    rows = cursor.fetchall()
    columns = [col[0] for col in cursor.description]
    conn.close()
    return [dict(zip(columns, r)) for r in rows]

if __name__ == "__main__":
    init_sqlite_db()
    init_faiss_rag()
