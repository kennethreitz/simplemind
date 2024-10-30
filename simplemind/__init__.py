from typing import List, Optional

from .models import Conversation, BasePlugin
from .utils import find_provider
from .settings import settings


def create_conversation(
    llm_model=None, llm_provider=None, *, plugins: Optional[List[BasePlugin]] = None
):
    """Create a new conversation."""

    # Create the conversation.
    conversation = Conversation(
        llm_model=llm_model, llm_provider=llm_provider or settings.DEFAULT_LLM_PROVIDER
    )

    # Add plugins to the conversation.
    for plugin in plugins or []:
        conversation.add_plugin(plugin)

    return conversation


def generate_data(prompt, *, llm_model=None, llm_provider=None, response_model=None):
    """Generate structured data from a given prompt."""

    # Find the provider.
    provider = find_provider(llm_provider or settings.DEFAULT_LLM_PROVIDER)

    # Generate the data.
    return provider.structured_response(
        prompt=prompt,
        llm_model=llm_model,
        response_model=response_model,
    )


def generate_text(prompt, *, llm_model=None, llm_provider=None, **kwargs):
    """Generate text from a given prompt."""

    # Find the provider.
    provider = find_provider(llm_provider or settings.DEFAULT_LLM_PROVIDER)

    # Generate the text.
    return provider.generate_text(prompt=prompt, llm_model=llm_model, **kwargs)


__all__ = [
    "create_conversation",
    "find_provider",
    "generate_data",
    "generate_text",
    "settings",
]
