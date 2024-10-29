from abc import ABC, abstractmethod

from instructor import Instructor


class BaseProvider(ABC):
    """The base provider class."""

    NAME: str
    DEFAULT_MODEL: str

    @property
    @abstractmethod
    def client(self):
        """The instructor client for the provider."""
        raise NotImplementedError

    @property
    @abstractmethod
    def structured_client(self) -> Instructor:
        """The structured client for the provider."""
        raise NotImplementedError

    @abstractmethod
    def send_conversation(self, conversation: "Conversation") -> "Message":
        """Send a conversation to the provider."""
        raise NotImplementedError

    @abstractmethod
    def structured_response(self, prompt: str, response_model, **kwargs):
        """Get a structured response."""
        raise NotImplementedError

    @abstractmethod
    def generate_text(self, prompt: str, **kwargs) -> str:
        """Generate text from a prompt."""
        raise NotImplementedError
