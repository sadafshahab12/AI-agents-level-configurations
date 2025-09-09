from agents import (
    Agent,
    AsyncOpenAI,
    OpenAIChatCompletionsModel,
    ModelSettings,
    Runner,
    set_tracing_disabled,
    function_tool,
)
import requests
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


weather_api_key = os.getenv("WEATHER_API_KEY")


@function_tool
def weather_tool(city):
    """Fetch the current weather for the given city"""
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
    weather_agent = Agent(
        name="Weather Teller",
        instructions="Fetch the realtime weather data of karachi.",
        model=model,
        tools=[weather_tool],
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
    result = await Runner.run(weather_agent, "Tell me today's weather in karachi.")
    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
