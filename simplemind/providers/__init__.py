from typing import List, Type

from ._base import BaseProvider
from .anthropic import Anthropic
from .gemini import Gemini
from .groq import Groq
from .openai import OpenAI
from .ollama import Ollama
from .xai import XAI

providers: List[Type[BaseProvider]] = [Anthropic, Gemini, Groq, OpenAI, Ollama, XAI]
