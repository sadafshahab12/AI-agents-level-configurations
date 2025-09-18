import asyncio
from dotenv import load_dotenv
import os
import random
from agents import (
    Agent,
    function_tool,
    Runner,
    AsyncOpenAI,
    OpenAIChatCompletionsModel,
    set_tracing_disabled,
    ItemHelpers,
)

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
set_tracing_disabled(True)

client = AsyncOpenAI(api_key=GEMINI_API_KEY, base_url=BASE_URL)
model = OpenAIChatCompletionsModel(model="gemini-2.0-flash", openai_client=client)


@function_tool
def how_many_jokes(count: int = None):
    """Return the number of jokes requested by user, otherwise random 1–5."""
    if count:
        return count
    return random.randint(1, 5)


async def main():
    agent = Agent(
        name="Smart Assistant",
        instructions="You are a helpful assistant. Always call the tool `how_many_jokes` "
        "to decide the number of jokes. If the user specifies a number, pass it as `count`. "
        "After getting the number from the tool, you **must generate that many jokes** "
        "and write them in a friendly list format.",
        model=model,
        tools=[how_many_jokes],
    )

    # print("Debug Context: ", ctx)
    result = Runner.run_streamed(agent, "Hello, tell me 4 jokes")
    # print(result)
    async for event in result.stream_events():
        # Filter: agar item hai to process karo
        if hasattr(event, "item"):
            if event.item.type == "tool_call_output_item":
                print(f"Tool Output: {event.item.output}")
            elif event.item.type == "message_output_item":
                print(ItemHelpers.text_message_output(event.item))
    else:
        # Debugging ke liye dekh lo konsa event aaya
        print("⚡ Meta Event:", type(event).__name__)

    # print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
