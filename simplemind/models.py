import uuid
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from .utils import find_provider


class SMBaseModel(BaseModel):
    date_created: datetime = Field(default_factory=datetime.now)

    def __str__(self):
        return f"<{self.__class__.__name__} {self.model_dump_json()}>"

    def __repr__(self):
        return str(self)


class BaseProvider(SMBaseModel):
    """The base provider class."""

    __name__ = "BaseProvider"
    DEFAULT_MODEL = "DEFAULT_MODEL"

    @property
    def client(self):
        """The instructor client for the provider."""
        raise NotImplementedError

    @property
    def structured_client(self):
        """The structured client for the provider."""
        raise NotImplementedError

    def send_conversation(self, conversation: "Conversation"):
        """Send a conversation to the provider."""
        raise NotImplementedError

    def structured_response(self, prompt: str, response_model, **kwargs):
        """Get a structured response."""
        raise NotImplementedError

    def generate_text(self, prompt: str, **kwargs):
        """Generate text from a prompt."""
        raise NotImplementedError


class BasePlugin(SMBaseModel):
    """The base plugin class."""


class Message(SMBaseModel):
    role: Literal["system", "user", "assistant"]
    text: str
    meta: Dict[str, Any] = {}
    raw: Optional[Any] = None
    llm_model: Optional[str] = None
    llm_provider: Optional[str] = None

    def __str__(self):
        return f"<Message role={self.role} text={self.text!r}>"

    @classmethod
    def from_raw_response(cls, *, text, raw):
        self = cls()
        self.text = text
        self.raw = raw
        return self


class Conversation(SMBaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    messages: List[Message] = []
    llm_model: Optional[str] = None
    llm_provider: Optional[str] = None
    plugins: List[Any] = []

    def __str__(self):
        return f"<Conversation id={self.id!r}>"

    def prepend_system_message(self, role: str, text: str, meta: Optional[Dict[str, Any]] = None):
        self.messages = [Message(role=role, text=text, meta=meta or {})] + self.messages
    def add_message(self, role: str, text: str, meta: Dict[str, Any] = {}):
        """Add a new message to the conversation."""
        self.messages.append(Message(role=role, text=text, meta=meta))

    def send(
        self, llm_model: Optional[str] = None, llm_provider: Optional[str] = None
    ) -> Message:
        """Send the conversation to the LLM."""
        for plugin in self.plugins:
            plugin.send_hook(self)

        provider = find_provider(llm_provider or self.llm_provider)
        response = provider.send_conversation(self)

        self.add_message(role="assistant", text=response.text, meta=response.meta)
        return response

    def add_plugin(self, plugin: Any):
        """Add a plugin to the conversation."""
        self.plugins.append(plugin)
