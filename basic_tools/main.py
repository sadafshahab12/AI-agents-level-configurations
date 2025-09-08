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
    ModelSettings,
)

load_dotenv()

set_tracing_disabled(True)

gemini_api_key = os.getenv("GEMINI_API_KEY")
BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

external_client = AsyncOpenAI(api_key=gemini_api_key, base_url=BASE_URL)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client,
)


@function_tool
def sum(a, b):
    """Exact addition (use this instead of guessing math)."""
    return a + b


@function_tool
def multiply(a, b):
    """Exact multiplication (use this instead of guessing math )"""
    return a * b


# agent = Agent(
#     name="Assistant",
#     instructions=(
#         "You are a helpfull assistant."
#         "Always use tool for math questions. Always follow DMAS rule (division, multiplication, addition, subtraction)."
#         "Explain answers clearly and briefly for beginners."
#     ),
#     model=model,
#     tools=[multiply, sum],
#     allow_parallel_tools=True,
# )

math_tutor_agent = Agent(
    name="Math Tutor",
    instructions="You are a precise math tutor. Always show your work step by step.",
    model=model,
    tools=[multiply, sum],
    model_settings=ModelSettings(temperature=0.1, tool_choice="required"),
)
# agent_creative = Agent(
#     name="Story Writer",
#     instructions="You are a creative story teller.",
#     model_settings=ModelSettings(temperature=0.9),
# )

creative_writer = Agent(
    name="Creative Writer",
    instructions="You are a creative story teller. Write engaging, imaginative stories. ",
    model=model,
    model_settings=ModelSettings(temperature=0.8, max_tokens=500),
)
prompt = "can you tell me something about math?"
prompt2 = "Write a short story about quaid-e-azam."
# result = Runner.run_sync(math_tutor_agent, prompt)
result2 = Runner.run_sync(creative_writer, prompt2)
# print(result.final_output)
print("\n Calling Agent \n")
print(result2.final_output)
