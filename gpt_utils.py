import os
import time
import requests
from dotenv import load_dotenv
from query_utils import get_similar_videos

# Load API Key from .env file
load_dotenv()
QWEN_API_KEY = os.getenv("DASHSCOPE_API_KEY")  
QWEN_API_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions"


def ask_qwen_max(question, retries=3):
    """Retrieve similar videos using pgvector and send user query + related video metadata to Qwen-Max API."""

    if not QWEN_API_KEY:
        return "❌ DASHSCOPE_API_KEY is missing. Please set it in your .env file."

    # Retrieve similar videos using semantic search
    related_videos = get_similar_videos(question)
    
    if not related_videos:
        return "⚠ No relevant videos found. Please refine your query."

    # Prepare metadata from related videos
    metadata = "\n".join(
        [f"- {v[1]} (ID: {v[0]}) | {v[3][:500]}..." for v in related_videos]  # v[3] is transcription
    )

    # Construct the prompt with clear instructions
    prompt = f"""
    User asked: "{question}"
    
    ## 🔍 **Similar Video References:**
    {metadata}
    
    ## 📌 **Your Role:**  
    You are an **expert AI assistant** with access to:
    1️⃣ **Metadata & transcripts** of similar videos from **pgvector**.  
    2️⃣ Your **own vast knowledge & training data**.  
    3️⃣ **Live internet data** (if available) to ensure real-time accuracy.  

    ## 🎯 **Your Objectives:**
    🔹 **1. Contextual Understanding:**  
       - Thoroughly analyze the **user query** in relation to the **provided video metadata** and your **prior knowledge**.  
       - Identify key **concepts, facts, and potential misunderstandings**.  

    🔹 **2. Multi-Layered Information Processing:**  
       - Cross-reference **transcripts & metadata** with your **knowledge**.  
       - Extract **core insights, useful references, and examples**.  

    🔹 **3. Structured & Engaging Communication:**  
       - Provide a **clear, engaging, and structured response**.  
       - Use **sections, bullet points, and step-by-step explanations** for clarity.  
       - If relevant, offer **analogies, examples, and historical context**.  

    🔹 **4. Dynamic Interaction & Guidance:**  
       - Anticipate possible **follow-up questions** and suggest related topics.  
       - Encourage further discussion by offering **additional insights or clarifications**.  
       - If the answer is complex, provide a **TL;DR summary** at the end.  

    ## 📢 **Final Instructions:**
    🚀 **Now, using all the available resources, generate the most insightful, structured, and comprehensive response possible.**  
    🎯 Keep it **concise yet informative**, **accurate yet engaging**.  
    - **Provide a well-structured, precise answer** that directly addresses the topic.  
    - **Avoid categorizing everything under AI unless explicitly relevant.**  
    - **Use bullet points**  
    """

    headers = {
        "Authorization": f"Bearer {QWEN_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "qwen-max",
        "messages": [
            {"role": "system", "content": "You are an AI assistant specialized in IT, Computer Science, and Data Science. Stay strictly on topic."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 4000
    }

    for attempt in range(retries):
        try:
            print(f"Attempt {attempt + 1} of {retries}...")
            response = requests.post(QWEN_API_URL, headers=headers, json=payload)
            response.raise_for_status()

            response_json = response.json()
            choices = response_json.get("choices", [])
            if choices:
                return choices[0].get("message", {}).get("content", "⚠ No content in response.")
            else:
                return "⚠ Unexpected API response format."

        except requests.exceptions.HTTPError as http_err:
            if response.status_code == 429:
                wait_time = (2 ** attempt) * 5
                print(f"⚠ Rate limit exceeded. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"❌ HTTP error: {http_err}")
                break

        except requests.exceptions.RequestException as req_err:
            print(f"❌ API request error: {req_err}")
            break

        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            break

    return "⚠ An error occurred while processing your request."
