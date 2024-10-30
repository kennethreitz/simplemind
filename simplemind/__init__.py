from typing import List, Optional, Type

from .models import Conversation, BasePlugin, BaseModel
from .utils import find_provider
from .settings import settings


class Session:
    """A session object that maintains configuration across multiple API calls.

    Similar to `requests.Session`, this allows you to specify default settings
    that will be used for all operations within the session.
    """

    def __init__(
        self,
        *,
        llm_provider: str = settings.DEFAULT_LLM_PROVIDER,
        llm_model: str = settings.DEFAULT_LLM_MODEL,
        **kwargs,
    ):
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.default_kwargs = kwargs

    def generate_text(self, prompt: str, **kwargs) -> str:
        """Generate text using the session's default provider and model."""
        merged_kwargs = {**self.default_kwargs, **kwargs}
        return generate_text(
            prompt=prompt,
            llm_provider=self.llm_provider,
            llm_model=self.llm_model,
            **merged_kwargs,
        )

    def generate_data(
        self, prompt: str, response_model: Type[BaseModel], **kwargs
    ) -> BaseModel:
        """Generate structured data using the session's default provider and model."""
        merged_kwargs = {**self.default_kwargs, **kwargs}
        return generate_data(
            prompt=prompt,
            response_model=response_model,
            llm_provider=self.llm_provider,
            llm_model=self.llm_model,
            **merged_kwargs,
        )

    def create_conversation(self, **kwargs) -> "Conversation":
        """Create a conversation using the session's default provider and model."""
        merged_kwargs = {**self.default_kwargs, **kwargs}
        return create_conversation(
            llm_provider=self.llm_provider, llm_model=self.llm_model, **merged_kwargs
        )


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
    "BasePlugin",
    "Session",
]
