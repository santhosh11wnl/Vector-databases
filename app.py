import os
import logging
from flask import Flask, request, jsonify, render_template
from gpt_utils import ask_qwen_max
from db_utils import setup_database
from dotenv import load_dotenv
from query_utils import get_similar_videos
from similarity import insert_video_data

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route('/')
def index():
    """Serve the frontend page."""
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    """API endpoint to handle user queries and return video results."""
    data = request.get_json()
    user_query = data.get("query", "").strip()

    if not user_query:
        logger.warning("⚠ Empty query received.")
        return jsonify({"error": "Query cannot be empty"}), 400

    try:
        logger.info(f"🔍 Processing query: {user_query}")

        # Fetch similar videos using vector search
        similar_videos = get_similar_videos(user_query)

        if not similar_videos:
            logger.info(f"📭 No similar videos found for query: {user_query}")
            return jsonify({
                "message": "No relevant videos found.",
                "ai_response": "No relevant data found for your query."
            })

        # Query Qwen-Max AI with user query and relevant videos
        qwen_response = ask_qwen_max(user_query, similar_videos)

        response_data = {
            "ai_response": qwen_response,
            "similar_videos": similar_videos
        }

        logger.info(f"✅ Query processed successfully: {user_query}")
        return jsonify(response_data)

    except Exception as e:
        logger.error(f"❌ Error processing query '{user_query}': {str(e)}", exc_info=True)
        return jsonify({"error": "An internal error occurred. Please try again later."}), 500

def initialize_database():
    """Ensure database is set up and video data is inserted before app starts."""
    try:
        logger.info("🔄 Setting up database and populating video data...")
        setup_database()
        insert_video_data()
        logger.info("✅ Database setup complete!")
    except Exception as e:
        logger.critical(f"❌ Database setup failed: {e}", exc_info=True)
        raise SystemExit("Database setup failed, stopping application.")

if __name__ == '__main__':
    # Initialize database before starting the app
    initialize_database()

    # Run the Flask app
    app.run(debug=True, host="0.0.0.0", port=5000, use_reloader=True)
