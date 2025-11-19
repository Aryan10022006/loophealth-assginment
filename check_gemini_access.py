import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('GOOGLE_API_KEY')

print("=" * 60)
print("CHECKING GOOGLE API KEY AND AVAILABLE MODELS")
print("=" * 60)
print(f"\nAPI Key: {api_key[:20]}..." if api_key else "API Key: NOT FOUND")

# Try different approaches to list models
print("\n\n1. Trying google.generativeai.list_models()...")
try:
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    
    models = list(genai.list_models())
    print(f"✅ Found {len(models)} models")
    
    for model in models[:10]:  # Show first 10
        try:
            print(f"  - {model.name}")
            if hasattr(model, 'supported_generation_methods'):
                print(f"    Methods: {model.supported_generation_methods}")
        except Exception as e:
            print(f"    Error reading model: {e}")
            
except Exception as e:
    print(f"❌ Error: {e}")

print("\n\n2. Testing specific model names...")
test_models = [
    'gemini-pro',
    'gemini-1.5-pro',
    'gemini-1.5-flash',
    'gemini-1.0-pro',
    'models/gemini-pro',
    'models/gemini-1.5-flash',
]

for model_name in test_models:
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content("Hi")
        print(f"✅ {model_name} - WORKS!")
        print(f"   Response: {response.text[:50]}...")
        break
    except Exception as e:
        error_msg = str(e)
        if '404' in error_msg:
            print(f"❌ {model_name} - Not found (404)")
        elif '429' in error_msg or 'quota' in error_msg.lower():
            print(f"⚠️  {model_name} - Quota exceeded")
        else:
            print(f"❌ {model_name} - {error_msg[:80]}")

print("\n" + "=" * 60)
print("TROUBLESHOOTING STEPS:")
print("=" * 60)
print("""
If all models show 404 errors, you need to:

1. Go to: https://aistudio.google.com/app/apikey
2. Check if your API key is valid
3. Go to: https://makersuite.google.com/app/prompts/new_freeform
4. Try the Gemini API in the web interface first
5. Make sure "Gemini API" is enabled in your Google Cloud Console

Alternative: Create a NEW API key:
1. Go to: https://aistudio.google.com/
2. Click "Get API key" 
3. Create new key or use existing project
4. Copy the new key to your .env file

Current API Key Status: Check usage at
https://ai.google.dev/gemini-api/docs/api-key
""")
