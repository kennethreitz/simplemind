from typing import List, Type

from ._base import BaseProvider
from ._base_tools import BaseTool
from .amazon import Amazon
from .anthropic import Anthropic
from .gemini import Gemini
from .groq import Groq
from .minimax import MiniMax, MiniMaxAnthropic
from .ollama import Ollama
from .openai import OpenAI
from .xai import XAI
from .deepseek import Deepseek

providers: List[Type[BaseProvider]] = [
    Anthropic,
    Gemini,
    Groq,
    OpenAI,
    Ollama,
    XAI,
    Amazon,
    Deepseek,
    MiniMax,
    MiniMaxAnthropic,
]

__all__ = [
    "Anthropic",
    "Gemini",
    "Groq",
    "OpenAI",
    "Ollama",
    "XAI",
    "Amazon",
    "providers",
    "BaseProvider",
    "BaseTool",
    "Deepseek",
    "MiniMax",
    "MiniMaxAnthropic",
]
