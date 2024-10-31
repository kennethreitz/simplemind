from typing import List, Type

from ._base import BaseProvider
from .anthropic import Anthropic
from .gemini import Gemini
from .groq import Groq
from .ollama import Ollama
from .openai import OpenAI
from .xai import XAI
from .amazon import Amazon

providers: List[Type[BaseProvider]] = [Anthropic, Gemini, Groq, OpenAI, Ollama, XAI, Amazon]
