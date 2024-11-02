from abc import ABC, abstractmethod
from functools import cached_property
from typing import TYPE_CHECKING, Any, Type, TypeVar

from instructor import Instructor
from pydantic import BaseModel

if TYPE_CHECKING:
    from ..models import Conversation, Message

T = TypeVar("T", bound=BaseModel)


class BaseProvider(ABC):
    """The base provider class."""

    NAME: str
    DEFAULT_MODEL: str
    supports_streaming: bool = False
    supports_structured_responses: bool = True

    @cached_property
    @abstractmethod
    def client(self) -> Any:
        """The instructor client for the provider."""
        raise NotImplementedError

    @cached_property
    @abstractmethod
    def structured_client(self) -> Instructor:
        """The structured client for the provider."""
        raise NotImplementedError

    @abstractmethod
    def send_conversation(self, conversation: "Conversation") -> "Message":
        """Send a conversation to the provider."""
        raise NotImplementedError

    @abstractmethod
    def structured_response(self, prompt: str, response_model: Type[T], **kwargs) -> T:
        """Get a structured response."""
        raise NotImplementedError

    @abstractmethod
    def generate_text(self, prompt: str, *, stream: bool = False, **kwargs) -> str:
        """Generate text from a prompt."""
        raise NotImplementedError
