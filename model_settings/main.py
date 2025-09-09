from agents import (
    Agent,
    AsyncOpenAI,
    OpenAIChatCompletionsModel,
    ModelSettings,
    Runner,
    set_tracing_disabled,
    function_tool,
)

import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")
BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
set_tracing_disabled(True)

client = AsyncOpenAI(api_key=gemini_api_key, base_url=BASE_URL)


@function_tool
def translate_story(story):
    """Translate the given story into Korean."""
    return f"Translate to Korean: {story}"


async def main():

    model = OpenAIChatCompletionsModel(model="gemini-2.0-flash", openai_client=client)

    focus_agent = Agent(
        name="chemistry tutor",
        instructions="You are chemistry tutor."
        "You will solve chemistry numericals only.",
        model=model,
        model_settings=ModelSettings(
            temperature=0.2,
        ),
    )

    creative_agent = Agent(
        name="Story Teller",
        instructions="Your are a story teller."
        "You will write creative stories that is never write or created.",
        model=model,
        tools=[translate_story],
        model_settings=ModelSettings(
            temperature=0.9,
            top_p=0.3,
        ),
    )
    prompt = """Write a short creative story about a cat in English. After writing the story, also provide its korean translation."""
    # prompt = """If 12 grams of carbon (C) completely react with 32 grams of oxygen (O₂) to form carbon dioxide (CO₂), calculate:
    # The mass of CO₂ formed.
    # The number of CO₂ molecules formed."""
    result = await Runner.run(creative_agent, prompt)
    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
