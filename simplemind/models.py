import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional


from pydantic import BaseModel, Field

from .utils import find_provider


MESSAGE_ROLE = Literal["system", "user", "assistant"]


class SMBaseModel(BaseModel):
    date_created: datetime = Field(default_factory=datetime.now)

    def __str__(self):
        return f"<{self.__class__.__name__} {self.model_dump_json()}>"

    def __repr__(self):
        return str(self)


class BasePlugin(ABC):
    """The base conversation plugin class."""

    @abstractmethod
    def send_hook(self, conversation: "Conversation"):
        """Send a hook to the plugin."""
        raise NotImplementedError


class Message(SMBaseModel):
    role: MESSAGE_ROLE
    text: str
    meta: Dict[str, Any] = {}
    raw: Optional[Any] = None
    llm_model: Optional[str] = None
    llm_provider: Optional[str] = None

    def __str__(self):
        return f"<Message role={self.role} text={self.text!r}>"

    @classmethod
    def from_raw_response(cls, *, text: str, raw):
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

    def add_message(
        self, role: MESSAGE_ROLE, text: str, meta: Optional[Dict[str, Any]] = None
    ):
        """Add a new message to the conversation."""
        if meta is None:
            meta = {}
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

    def get_last_message(self, role: MESSAGE_ROLE) -> Optional[Message]:
        """Get the last message with the given role."""
        return next((m for m in reversed(self.messages) if m.role == role), None)

    def add_plugin(self, plugin: Any):
        """Add a plugin to the conversation."""
        self.plugins.append(plugin)
