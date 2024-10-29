from .models import Conversation
from .utils import find_provider
from .settings import settings


def create_conversation(llm_model=None, llm_provider=None):
    """Create a new conversation."""

    return Conversation(
        llm_model=llm_model, llm_provider=llm_provider or settings.DEFAULT_LLM_PROVIDER
    )


def generate_data(prompt, *, llm_model=None, llm_provider=None, response_model=None):
    """Generate structured data from a given prompt."""

    provider = find_provider(llm_provider or settings.DEFAULT_LLM_PROVIDER)

    return provider.structured_response(
        prompt=prompt,
        llm_model=llm_model,
        response_model=response_model,
    )


def generate_text(prompt, *, llm_model=None, llm_provider=None, **kwargs):
    """Generate text from a given prompt."""
    provider = find_provider(llm_provider or settings.DEFAULT_LLM_PROVIDER)

    return provider.generate_text(prompt=prompt, llm_model=llm_model, **kwargs)


__all__ = [
    "Conversation",
    "create_conversation",
    "find_provider",
    "generate_data",
    "generate_text",
    "settings",
]
