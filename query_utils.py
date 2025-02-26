from similarity import connect_db, generate_embedding

def get_similar_videos(user_query, top_k=5):
    """Retrieve similar videos using pgvector similarity search."""
    
    query_embedding = generate_embedding(user_query)
    conn = connect_db()
    cur = conn.cursor()

    cur.execute("""
        SELECT id, name, metadata, transcription, 1 - (transcription_embedding <=> %s::vector) AS similarity
        FROM videos
        ORDER BY similarity DESC
        LIMIT %s;
    """, (query_embedding, top_k))

    results = cur.fetchall()
    conn.close()

    return results
