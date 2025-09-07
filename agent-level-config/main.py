from openai import AsyncOpenAI

from agents import Agent, Runner, OpenAIChatCompletionsModel
import os
from dotenv import load_dotenv
import asyncio

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")


async def main():
    client = AsyncOpenAI(
        api_key=gemini_api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )
    model = OpenAIChatCompletionsModel(model="gemini-2.0-flash", openai_client=client)
    history_agent = Agent(
        name="history_agent",
        handoff_description="Specialist agent for history questions.",
        instructions="You provide assistance with historical queries. Explain important events and context clearly.",
        model=model,
    )
    math_agent = Agent(
        name="coding tutor",
        handoff_description="Specialist agent for coding questions.",
        instructions="You provide assistance with coding questions. Explain concepts clearly and provide step-by-step solutions.",
        model=model,
    )
    triage_agent = Agent(
        name="triage_agent",
        instructions="You are a triage agent. Your job is to determine the appropriate specialist agent to handle a user's request. You have access to the following specialist agents.",
        handoffs=[history_agent, math_agent],
        model=model,
    )

    result = await Runner.run(triage_agent, "What is 5 + 2?")

    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
