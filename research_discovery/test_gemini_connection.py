"""
test_gemini_connection.py
─────────────────────────
Verifies the Gemini API connection using google-genai (new SDK).
Does NOT print or expose the API key.

Run: python test_gemini_connection.py
"""

import os
import sys
from pathlib import Path

# Load .env from the project root
project_root = Path(__file__).resolve().parent
env_file = project_root / ".env"

if not env_file.exists():
    print("❌ .env file not found at:", env_file)
    print("   Run: copy .env.example .env  (then add your GEMINI_API_KEY)")
    sys.exit(1)

from dotenv import load_dotenv
load_dotenv(env_file, override=True)

api_key = os.getenv("GEMINI_API_KEY", "").strip().strip('"').strip("'")
model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash").strip()

if not api_key:
    print("❌ GEMINI_API_KEY is not set in .env")
    print("   Format: GEMINI_API_KEY=your_key_here   (no quotes, no spaces)")
    sys.exit(1)

# Show only first 8 and last 4 chars to confirm the key was read (safe)
masked = api_key[:8] + "..." + api_key[-4:]
print(f"✅ GEMINI_API_KEY loaded  (masked: {masked})")
print(f"   Model : {model_name}")
print(f"   SDK   : google-genai (new SDK)")
print()

# Try importing google-genai (new SDK)
try:
    from google import genai
    print("✅ google-genai imported successfully")
except ImportError:
    print("❌ google-genai not installed.")
    print("   Run: pip install google-genai")
    sys.exit(1)

# Configure client and make a minimal test call
print("\n⏳ Sending test prompt to Gemini API...")
try:
    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model=model_name,
        contents="Reply with exactly one sentence: 'PaperGraph Gemini connection verified.'",
        config={"automatic_function_calling": {"disable": True}},
    )

    reply = response.text.strip() if response.text else "(empty response)"
    print(f"\n✅ Gemini API responded successfully!")
    print(f"   Model response: {reply}")
    print(f"\n🎉 API connection is working. PaperGraph LLM stages are ready.")

except Exception as exc:
    err = str(exc)
    print(f"\n❌ Gemini API call failed: {err}")
    print()

    if "404" in err or "not found" in err.lower():
        print("💡 Model not found. Try setting in .env:")
        print("      GEMINI_MODEL=gemini-2.0-flash")
        print("   or GEMINI_MODEL=gemini-1.5-flash-latest")
        print("   or GEMINI_MODEL=gemini-1.5-pro-latest")
        print()
        # Try to list available models
        try:
            print("   Available models with generateContent support:")
            for m in client.models.list():
                if hasattr(m, 'supported_actions') and 'generateContent' in str(m.supported_actions):
                    print(f"      {m.name}")
                elif hasattr(m, 'name') and 'gemini' in str(m.name).lower():
                    print(f"      {m.name}")
        except Exception:
            pass
    elif "API_KEY" in err or "key" in err.lower() or "auth" in err.lower():
        print("💡 API key issue — check your key at https://aistudio.google.com/app/apikey")
        print("   Make sure the Generative Language API is enabled in your Google Cloud project.")
    elif "quota" in err.lower():
        print("💡 Quota exceeded — wait a moment or check your project billing.")

    sys.exit(1)
