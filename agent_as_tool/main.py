from agents import (
    AsyncOpenAI,
    Runner,
    Agent,
    OpenAIChatCompletionsModel,
    set_tracing_disabled,
    ModelSettings
)
import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")
BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

set_tracing_disabled(True)
client = AsyncOpenAI(api_key=gemini_api_key, base_url=BASE_URL)
model = OpenAIChatCompletionsModel(model="gemini-2.0-flash", openai_client=client)


async def main():
    spanish_agent = Agent(
        name="Spanish Agent",
        instructions="You translate the user's message to spanish",
        model=model,
    )
    french_agent = Agent(
        name="French Agent",
        instructions="You translate the user's message to French.",
        model=model,
    )
    orchestrator_agent = Agent(
        name="orchestrator agent",
        instructions=(
            "You are a translation agent. You use the tool given to you to translate."
            "If asked for multiple translations, you call the relevant tools."
        ),
        model=model,
        tools=[
            spanish_agent.as_tool(
                tool_name="translate_to_spanish",
                tool_description="Translate the user's message to Spanish.",
            ),
            french_agent.as_tool(
                tool_name="translate_to_french",
                tool_description="Translate the user's message to French.",
            ),
        ],
        model_settings=ModelSettings(
            tool_choice="required",
        )
    )

    result = await Runner.run(
        orchestrator_agent, input="Say 'hello how are you'"
    )
    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
