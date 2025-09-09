from agents import Agent, AsyncOpenAI, OpenAIChatCompletionsModel, ModelSettings, Runner, set_tracing_disabled

import asyncio
import os
from dotenv import load_dotenv
load_dotenv()

gemini_api_key=os.getenv()