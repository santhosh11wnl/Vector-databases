import os
from similarity import insert_video_data
from gpt_utils import ask_qwen_max
from db_utils import setup_database

def process_user_query(user_query):
    """Call Qwen-Max directly since it handles fetching similar videos."""
    try:
        qwen_response = ask_qwen_max(user_query)  
    except Exception as e:
        print(f"\n⚠ Error getting AI response: {str(e)}")
        return None
    print("\n🧠 **AI Response:**\n")
    print(qwen_response)
    return qwen_response

def main():
    """Interactive chatbot to handle user queries based on video transcripts."""
    print("\n💡 **Welcome to the AI Video Query System!**")
    print("🔹 Ask a question based on available video transcripts.")
    print("🔹 Type **'exit'**, **'quit'**, or **'q'** to end the session.\n")

    while True:
        user_query = input("📝 **You:** ").strip()
        if user_query.lower() in ["exit", "quit", "q"]:
            print("\n👋 **Exiting... Have a great day! 🚀**")
            break  

        if not user_query:
            print("\n⚠ **Please enter a valid query!**")
            continue

        print("\n⏳ **Processing your request...**\n")
        process_user_query(user_query)
        print("\n────────────────────────────────────────\n")

if __name__ == "__main__":
    try:
        setup_database()  # Ensure the database and pgvector extension are ready
        insert_video_data()  # Populate database from transcriptions
    except Exception as e:
        print(f"\n⚠ **Setup error:** {str(e)}")
    else:
        main()  # Start the chatbot
