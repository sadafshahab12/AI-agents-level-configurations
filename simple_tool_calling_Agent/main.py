from agents import (
    AsyncOpenAI,
    Runner,
    Agent,
    OpenAIChatCompletionsModel,
    ModelSettings,
    set_tracing_disabled,
    function_tool,
)
import asyncio
import os
import requests
from dotenv import load_dotenv

load_dotenv()
set_tracing_disabled(True)
gemini_api_key = os.getenv("GEMINI_API_KEY")
BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
client = AsyncOpenAI(api_key=gemini_api_key, base_url=BASE_URL)
model = OpenAIChatCompletionsModel(model="gemini-2.0-flash", openai_client=client)

joke_api_url = "https://v2.jokeapi.dev/joke/Any?blacklistFlags=religious,political,racist,explicit&amount=5"


@function_tool
def joker_tool(amount):
    """This tool is for joke telling. It will return 5 random jokes"""
    url = f"https://v2.jokeapi.dev/joke/Any?amount={amount}"
    response = requests.get(url)
    data = response.json()

    jokes = data["jokes"]
    
    main_jokes=[]
    for joke in jokes:
        if joke["type"] == "single":
            main_jokes.append(f"{joke["joke"]} (category: {joke["category"]})")
        elif joke["type"] == "twopart":
            main_jokes.append(f"{joke["setup"]} {joke["delivery"]} (Category:{joke["category"]})")
            
    return "\n".join(main_jokes)



joke_agent = Agent(
    name="Funny joker",
    instructions="You are a funny agent. Always use the joker tool to tell jokes when user asks.",
    model=model,
    tools=[joker_tool],
)


async def main():
    result = await Runner.run(joke_agent, input="Tell me 4 jokes")
    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
