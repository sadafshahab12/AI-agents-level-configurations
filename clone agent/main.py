import asyncio
from dotenv import load_dotenv
import os
from agents import (
    Agent,
    Runner,
    AsyncOpenAI,
    OpenAIChatCompletionsModel,
    set_tracing_disabled,
    ModelSettings,
)

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
set_tracing_disabled(True)

client = AsyncOpenAI(api_key=GEMINI_API_KEY, base_url=BASE_URL)
model = OpenAIChatCompletionsModel(model="gemini-2.0-flash", openai_client=client)

base_agent = Agent(
    name="BaseAssistant",
    instructions="You are angry assistant.",
    model=model,
    model_settings=ModelSettings(temperature=0.5, max_tokens=50),
)
friendly_agent = base_agent.clone(
    name="Friendly Assistant",
    instructions="You are friendly and lovely assistant.",
    model=model,
    model_settings=ModelSettings(temperature=0.9),
)
creative_agent = base_agent.clone(
    name="CreativeAssistant",
    instructions="You are creative assistant.",
    model_settings=ModelSettings(temperature=0.9, max_tokens=50),  # more creativity
)
precise_agent = base_agent.clone(
    name="PreciseAssistant",
    instructions="You are creative assistant.",
    model_settings=ModelSettings(temperature=0.1, max_tokens=50),  # more creativity
)

base_agent_result = Runner.run_sync(base_agent, "Describe a sunset.")
friendly_agent_result = Runner.run_sync(friendly_agent, "Hello How are you?")
creative_agent_result = Runner.run_sync(
    creative_agent, "give me gifting idea what should i sell?"
)
precise_agent_result = Runner.run_sync(precise_agent, "Describe a sunset.")


print(base_agent_result.final_output)
print(friendly_agent_result.final_output)
print(creative_agent_result.final_output)
print(precise_agent_result.final_output)
