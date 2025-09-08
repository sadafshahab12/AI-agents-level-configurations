import os
from dotenv import load_dotenv

from agents import (
    AsyncOpenAI,
    Runner,
    OpenAIChatCompletionsModel,
    Agent,
    function_tool,
    set_default_openai_client,
    set_tracing_disabled,
)

load_dotenv()

set_tracing_disabled(True)

gemini_api_key = os.getenv("GEMINI_API_KEY")
BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

external_client = AsyncOpenAI(api_key=gemini_api_key, base_url=BASE_URL)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash", openai_client=external_client
)

@function_tool
def sum(a, b):
    """Exact addition (use this instead of guessing math)."""
    return a + b

@function_tool
def multiply(a, b):
    """Exact multiplication (use this instead of guessing math )"""
    return a * b


agent = Agent(
    name="Assistant",
    instructions=(
        "You are a helpfull assistant."
        "Always use tool for math questions. Always follow DMAS rule (division, multiplication, addition, subtraction)."
        "Explain answers clearly and briefly for beginners."
    ),
    model=model,
    tools=[multiply, sum],
)


prompt = "what is 19 + 23 * 2?"

result = Runner.run_sync(agent, prompt)

print("\n Calling Agent \n")
print(result.final_output)