from abc import ABC, abstractmethod
from functools import cached_property
from typing import TYPE_CHECKING, Any, Callable, Type, TypeVar

from instructor import Instructor
from pydantic import BaseModel

from simplemind.providers._base_tools import BaseTool

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
    def send_conversation(
        self,
        conversation: "Conversation",
        tools: list[Callable | BaseTool] | None = None,
    ) -> "Message":
        """Send a conversation to the provider."""
        raise NotImplementedError

    @abstractmethod
    def structured_response(
        self, prompt: str, response_model: Type[T], **kwargs
    ) -> T:
        """Get a structured response."""
        raise NotImplementedError

    @abstractmethod
    def generate_text(
        self,
        prompt: str,
        *,
        tools: list[Callable | BaseTool] | None = None,
        stream: bool = False,
        **kwargs,
    ) -> str:
        """Generate text from a prompt."""
        raise NotImplementedError

    @cached_property
    @abstractmethod
    def tool(self) -> Type[BaseTool]:
        """The tool implementation for the provider."""
        raise NotImplementedError

    def make_tools(self, tools: list[Callable | BaseTool] | None):
        if tools is not None:
            return [self.tool.from_function(func) for func in tools]
        else:
            return []
