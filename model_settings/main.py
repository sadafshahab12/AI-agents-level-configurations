from agents import (
    Agent,
    AsyncOpenAI,
    OpenAIChatCompletionsModel,
    ModelSettings,
    Runner,
    set_tracing_disabled,
    function_tool,
)
from typing import List
import requests
import asyncio
import os
from dotenv import load_dotenv
from pydantic import BaseModel


class CalenderEvent(BaseModel):
    name: str
    date: str
    participants: list[str]


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

    calender_agent = Agent(
        name="Calender Extractor",
        instructions="You are a Calendar Event Extractor. Your job is to read and Extract all the calender events from the given data.",
        output_type=List[CalenderEvent],# multiple objects
        model=model,
    )
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
    prompt2 = """Upcoming events for the month:

1. Marketing Team Sync on 2025-09-10 with Alice, Bob, and Charlie to discuss new campaign strategies.
2. Product Launch Meeting on 2025-09-12 with Dana, Evan, Frank, and Grace at the main conference room.
3. Client Presentation for ACME Corp on 2025-09-15 with Henry and Irene; prepare slides and reports.
4. Design Workshop on 2025-09-18 with Julia, Kevin, and Laura focusing on UI/UX improvements.
5. Finance Review Meeting on 2025-09-20 with Michael, Nancy, and Olivia to finalize quarterly budget.
6. Team Building Activity on 2025-09-22 with all department members, location: Central Park.
7. Board Meeting on 2025-09-25 with Peter, Quinn, Rachel, and Steve to approve next quarter plans.
8. Training Session on 2025-09-28 with Tony, Uma, and Victor about new software tools.
"""
    result2 = await Runner.run(calender_agent, prompt2)
    print(result2.final_output)


if __name__ == "__main__":
    asyncio.run(main())
