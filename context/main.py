import asyncio
from dataclasses import dataclass
from dotenv import load_dotenv
import os
from agents import (
    Agent,
    RunContextWrapper,
    Runner,
    function_tool,
    AsyncOpenAI,
    OpenAIChatCompletionsModel,
    set_tracing_disabled,
)

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
set_tracing_disabled(True)


# define a context using a data class
@dataclass
class UserInfo:
    name: str
    uid: int


# a tool function that accesses local context via the wrapper
@function_tool
async def fetch_user_age(wrapper: RunContextWrapper[UserInfo]):
    return f"User {wrapper.context.name} is 47 years old."


weather_api_key = os.getenv("WEATHER_API_KEY")


@function_tool
def fetch_weather_tool(city):
    """fetch the weather"""
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={weather_api_key}&units=metric"


async def main():
    # create your context object
    user_info = UserInfo(name="john", uid=123)
    client = AsyncOpenAI(api_key=GEMINI_API_KEY, base_url=BASE_URL)
    model = OpenAIChatCompletionsModel(model="gemini-2.0-flash", openai_client=client)
    agent = Agent[UserInfo](
        name="Assistant",
        tools=[fetch_user_age],
        instructions=f"The user’s name is {user_info.name}, and their ID is {user_info.uid}. "
        f"If asked about age, you may use the fetch_user_age tool.",
        model=model,
    )

    result = await Runner.run(
        starting_agent=agent,
        input="Who is the user?",
        context=user_info,
    )

    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
