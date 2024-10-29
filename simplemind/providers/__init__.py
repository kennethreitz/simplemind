from typing import List, Type
from .anthropic import Anthropic
from .groq import Groq
from .openai import OpenAI
from .xai import XAI
from ._base import BaseProvider

providers: List[Type[BaseProvider]] = [Anthropic, Groq, OpenAI, XAI]
