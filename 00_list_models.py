"""Print the Gemini models your API key can call, so you can set TEXT_MODEL / VIDEO_MODEL."""
import os
from google import genai
c = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
for m in c.models.list():
    if "generateContent" in (m.supported_actions or []):
        print(m.name.replace("models/", ""))
