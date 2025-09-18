import asyncio
from dotenv import load_dotenv
import os
from agents import (
    Agent,
    RunContextWrapper,
    Runner,
    AsyncOpenAI,
    OpenAIChatCompletionsModel,
    set_tracing_disabled,
)

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
set_tracing_disabled(True)

client = AsyncOpenAI(api_key=GEMINI_API_KEY, base_url=BASE_URL)
model = OpenAIChatCompletionsModel(model="gemini-2.0-flash", openai_client=client)


class UserContext:
    def __init__(self, name: str):
        self.name = name


def dynamic_instructions(context: RunContextWrapper[UserContext], agent: Agent) -> str:
    # debugging
    user_name = context.context.context.name
    # print("Context wrapper: ", context.context)
    # print("inner: ", inner.__dict__)
    print("User: ", user_name)
    return (
        f"You are {agent.name}. The user's name is {user_name}. "
        f"Always greet them by their name and personalize answers. "
        f"Your work is to help {user_name} clearly and politely."
    )


def context_aware(context: RunContextWrapper, agent: Agent) -> str:
    message_count = len(getattr(context, "messages", []))

    if message_count == 0:
        return "You are a welcoming assistant. Introduce yourself!"
    elif message_count < 3:
        return "You are a helpful assistant. Be encouraging and detailed."
    else:
        return "You are an experienced assistant. Be concise but thorough."


agent = Agent[UserContext](
    name="Smart Assistant",
    instructions=dynamic_instructions,
    model=model,
)
context_agent = Agent(
    name="Context Aware Agent", instructions=context_aware, model=model
)


async def main():
    ctx = RunContextWrapper(context=UserContext("Sadaf"))
    # print("Debug Context: ", ctx)
    result = await Runner.run(context_agent, "Hello, now tell me my name")
    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
