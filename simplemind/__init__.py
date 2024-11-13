import inspect
from typing import Callable, List, Type

from .models import BaseModel, BasePlugin, Conversation
from .settings import settings
from .utils import find_provider


class Session:
    """A session object that maintains configuration across multiple API calls.

    Similar to `requests.Session`, this allows you to specify default settings
    that will be used for all operations within the session.
    """

    def __init__(
        self,
        *,
        llm_provider: str = settings.DEFAULT_LLM_PROVIDER,
        llm_model: str | None = None,
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

    def create_conversation(self, **kwargs) -> Conversation:
        """Create a conversation using the session's default provider and model."""
        merged_kwargs = {**self.default_kwargs, **kwargs}
        return create_conversation(
            llm_provider=self.llm_provider, llm_model=self.llm_model, **merged_kwargs
        )


def create_conversation(
    *,
    llm_model: str | None = None,
    llm_provider: str | None = None,
    plugins: List[BasePlugin] | None = None,
    **kwargs,
) -> Conversation:
    """Create a new conversation."""

    # Create the conversation.
    conv = Conversation(
        llm_model=llm_model,
        llm_provider=llm_provider or settings.DEFAULT_LLM_PROVIDER,
    )

    # Add plugins to the conversation.
    for plugin in plugins or []:
        conv.add_plugin(plugin)

    return conv


def generate_data(
    prompt: str,
    *,
    llm_model: str | None = None,
    llm_provider: str | None = None,
    response_model: Type[BaseModel],
    **kwargs,
) -> BaseModel:
    """Generate structured data from a given prompt."""

    # Find the provider.
    provider = find_provider(llm_provider or settings.DEFAULT_LLM_PROVIDER)

    # Generate the data.
    return provider.structured_response(
        prompt=prompt,
        llm_model=llm_model,
        response_model=response_model,
        **kwargs,
    )


def generate_text(
    prompt: str,
    *,
    llm_model: str | None = None,
    llm_provider: str | None = None,
    stream: bool = False,
    **kwargs,
) -> str:
    """Generate text from a given prompt."""

    # Find the provider.
    provider = find_provider(llm_provider or settings.DEFAULT_LLM_PROVIDER)

    # Generate the text.
    if stream:
        if not provider.supports_streaming:
            raise ValueError(f"{provider} does not support streaming.")

        return provider.generate_stream_text(
            prompt=prompt, llm_model=llm_model, **kwargs
        )
    else:
        return provider.generate_text(prompt=prompt, llm_model=llm_model, **kwargs)


def enable_logfire() -> None:
    """Enable logfire logging."""
    settings.logging.enable_logfire()

def tool(
    llm_provider: str | None = None,
    llm_model: str | None = None,
):
    provider = find_provider(llm_provider or settings.DEFAULT_LLM_PROVIDER)

    def decorator(func: Callable):
        sig = inspect.signature(func)
        res = generate_data(
            (
                "Based on this function signature, fill up the required fieds."
                f"\nSignature: {func.__name__}{sig}"
                "Make sure to properly add the required field in `required` if there are no defaults"
            ),
            llm_provider=llm_provider,
            response_model=provider.tool,
        )
        res.raw_func = func
        res.__signature__ = sig
        res.__doc__ = func.__doc__

        return res

    return decorator

# Syntax sugar.
Plugin = BasePlugin

__all__ = [
    "create_conversation",
    "find_provider",
    "generate_data",
    "generate_text",
    "settings",
    "BasePlugin",
    "Session",
    "Plugin",
    "enable_logfire",
    "tool"
]
