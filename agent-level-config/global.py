from agents import (
    Agent,
    Runner,
    AsyncOpenAI,
    set_default_openai_client,
    set_tracing_disabled,
    set_default_openai_api,
)
import os
from dotenv import load_dotenv

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")
set_tracing_disabled(True)
set_default_openai_api("chat_completions")

external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)
set_default_openai_client(external_client)

agent = Agent(
    name="plastic manufacturing assistant",
    instructions="You are a helpful and friendly assistant for students in the plastic manufacturing field. Always explain concepts in simple, easy-to-understand language. Keep answers short, clear and to the point, while maintaining a supportive tone. You are just plastic manufacturer agent You only answer relevant question. Don't answer irrelevant questions.",
    model="gemini-2.0-flash",
)

result = Runner.run_sync(agent, "What are the weather?")

print(result.final_output)
