import asyncio
from dataclasses import dataclass
from dotenv import load_dotenv
import os
import requests
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
    """fetch the weather of the city"""
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={weather_api_key}&units=metric"

    response = requests.get(url)
    if response.status_code != 200:
        return f"Could not fetch weather for {city}."
    data = response.json()
    description = data["weather"][0]["description"]
    temp = data["main"]["temp"]
    humidity = data["main"]["humidity"]
    return f"Weather in {city} is {description} with temperature {temp} and humidity {humidity}%."


async def main():
    # create your context object
    user_info = UserInfo(name="john", uid=123)
    client = AsyncOpenAI(api_key=GEMINI_API_KEY, base_url=BASE_URL)
    model = OpenAIChatCompletionsModel(model="gemini-2.0-flash", openai_client=client)
    agent = Agent[UserInfo](
        name="Assistant",
        instructions="You must answer all parts of the user's query. Use tools if necessary.",
        model=model,
        tools=[fetch_weather_tool],
        tool_use_behavior="stop_on_first_tool",
    )

    result = await Runner.run(
        starting_agent=agent,
        input="Tell me a short joke and also the weather in Karachi.",
        max_turns=3,
    )

    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
