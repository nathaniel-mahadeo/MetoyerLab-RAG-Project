# test_api.py - CORRECTED IMPORT
# 🚨 THIS LINE WAS LIKELY MISSING OR WRONG:
from google import genai 

# 🔑 HARDCODE THE API KEY DIRECTLY INTO THE CLIENT
API_KEY = "Your_API_key_here"

# Pass the API key explicitly to the client constructor
client = genai.Client(api_key=API_KEY)

response = client.models.generate_content(
    model="gemini-2.5-flash", 
    contents="Explain how AI works in a few words"
)

print(response.text)