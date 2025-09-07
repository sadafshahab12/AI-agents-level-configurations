import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

model = genai.GenerativeModel("gemini-2.0-flash")

conversation = []

user_input = "Search the latest AI news in healthcare."

response1 = model.generate_content(user_input)

conversation.append({"role": "user", "content": user_input})
conversation.append({"role": "assistant", "content": response1.text})

print("Response 1:", response1.text)

follow_up = "Summarize those findings in bullet points."

history_text = "\n".join(
    [f"{m["role"].capitalize()} : {m["content"]}" for m in conversation]
)

prompt = history_text + f"\nUser: {follow_up}"

response2 = model.generate_content(prompt)

conversation.append({"role": "user", "content": follow_up})
conversation.append({"role": "assistant", "content": response2.text})

print("Response 2:", response2.text)
