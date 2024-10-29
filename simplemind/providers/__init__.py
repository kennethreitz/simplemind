from typing import List, Type

from simplemind.providers._base import BaseProvider
from simplemind.providers.anthropic import Anthropic
from simplemind.providers.groq import Groq
from simplemind.providers.openai import OpenAI
from simplemind.providers.ollama import Ollama
from simplemind.providers.xai import XAI

providers: List[Type[BaseProvider]] = [Anthropic, Groq, OpenAI, Ollama, XAI]
