import os
import psycopg2
import numpy as np
from dotenv import load_dotenv
from psycopg2.extras import Json
from sentence_transformers import SentenceTransformer

load_dotenv()

def connect_db():
    """Establish a connection to Neon DB."""
    return psycopg2.connect(
        dbname=os.getenv("NEON_DB"),
        user=os.getenv("NEON_USER"),
        password=os.getenv("NEON_PASSWORD"),
        host=os.getenv("NEON_HOST")
    )

# Load embedding model
model_path = r"C:\Users\saimi\.cache\huggingface\hub\models--sentence-transformers--all-mpnet-base-v2\snapshots\9a3225965996d404b775526de6dbfe85d3368642"

# Load model from local path
model = SentenceTransformer(model_path)

def generate_embedding(text):
    """Generate an embedding for a given text."""
    embedding = model.encode(text, normalize_embeddings=True)
    return list(map(float, embedding))  # Convert NumPy array to list

def setup_database():
    """Ensure pgvector is enabled and the database structure is created."""
    conn = connect_db()
    cur = conn.cursor()

    try:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        conn.commit()
        print("✅ pgvector extension enabled.")

        cur.execute("""
            CREATE TABLE IF NOT EXISTS videos (
                id SERIAL PRIMARY KEY,
                name TEXT NOT NULL,
                metadata JSONB,
                transcription TEXT NOT NULL,
                transcription_embedding VECTOR(768)
            );
        """)
        conn.commit()
        print("✅ Table 'videos' created successfully.")

    except psycopg2.Error as e:
        print(f"❌ Error setting up database: {e}")

    finally:
        cur.close()
        conn.close()

