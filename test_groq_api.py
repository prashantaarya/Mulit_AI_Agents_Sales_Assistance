# Create: test_groq_api.py
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

# Get API key
api_key = os.getenv('GROQ_API_KEY')
model_name = os.getenv('MODEL_NAME', 'llama-3.3-70b-versatile')

print(f"🔑 Testing API Key: {api_key[:10]}...")
print(f"🤖 Using Model: {model_name}")

try:
    # Initialize GROQ LLM
    llm = ChatGroq(
        temperature=0.1,
        model_name=model_name,
        groq_api_key=api_key  # Explicitly pass the API key
    )
    
    # Test with a simple query
    print("🧪 Testing API connection...")
    response = llm.invoke("Say hello in one word")
    print(f"✅ API Test SUCCESS: {response.content}")
    
except Exception as e:
    print(f"❌ API Test FAILED: {e}")
    
    # Additional debugging
    if "401" in str(e) or "invalid_api_key" in str(e):
        print("🔍 This is definitely an API key issue")
        print("💡 Solutions:")
        print("   1. Check if your API key is correct")
        print("   2. Verify your GROQ account has credits")
        print("   3. Make sure the API key isn't expired")
        print("   4. Try generating a new API key from GROQ console")
    elif "model" in str(e).lower():
        print("🔍 This might be a model name issue")
        print("💡 Try using: 'llama3-70b-8192' instead")
    else:
        print(f"🔍 Unexpected error: {type(e).__name__}")