from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel

from agents.run import RunConfig
import os
from dotenv import load_dotenv

load_dotenv()

external_client = AsyncOpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash", openai_client=external_client
)

config = RunConfig(model=model, model_provider=external_client, tracing_disabled=True)

agent = Agent(
    name="Recipe Assistant",
    instructions="You are a helpful assistant that provides recipes based on user input. The recipe should be in bullet points.",
)
result = Runner.run_sync(
    agent,
    "Can you provide me with a recipe for chocolate chip cookies?",
    run_config=config,
)
print(result.final_output)

