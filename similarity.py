from psycopg2.extras import Json
from db_utils import connect_db, generate_embedding
import os
import psycopg2

def read_transcription(file_path):
    """Read a transcription from a text file."""
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read().strip()

def insert_video_data():
    """Extract transcriptions from files, generate embeddings, and store in the database."""
    conn = connect_db()
    cur = conn.cursor()

    transcription_folder = os.getenv("TRANSCRIPTIONS_FOLDER")
    if not transcription_folder:
        print("❌ TRANSCRIPTIONS_FOLDER path missing in .env file.")
        return

    transcription_files = [os.path.join(transcription_folder, f) for f in os.listdir(transcription_folder) if f.endswith(".txt")]
    categories = ["Java"] * len(transcription_files)
    durations = ["Unknown"] * len(transcription_files)

    try:
        for i, file_path in enumerate(transcription_files):
            transcription_text = read_transcription(file_path)
            transcription_embedding = generate_embedding(transcription_text)

            video_name = os.path.basename(file_path).replace(".txt", ".mp4")
            metadata = {"category": categories[i], "duration": durations[i]}

            cur.execute("""
                INSERT INTO videos (name, metadata, transcription, transcription_embedding)
                VALUES (%s, %s, %s, %s)
                RETURNING id;
            """, (video_name, Json(metadata), transcription_text, transcription_embedding))

        conn.commit()
        print("✅ Transcriptions and embeddings stored successfully!")

    except psycopg2.Error as e:
        print(f"❌ Error inserting video data: {e}")

    finally:
        cur.close()
        conn.close()
