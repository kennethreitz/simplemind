import uuid
from datetime import datetime
from types import TracebackType
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from .utils import find_provider

MESSAGE_ROLE = Literal["system", "user", "assistant"]


class SMBaseModel(BaseModel):
    """The base SimpleMind model class."""

    date_created: datetime = Field(default_factory=datetime.now)

    def __str__(self):
        return f"<{self.__class__.__name__} {self.model_dump_json()}>"

    def __repr__(self):
        return str(self)


class BasePlugin(SMBaseModel):
    """The base conversation plugin class."""

    # Plugin metadata.
    meta: Dict[str, Any] = {}

    class Config:
        extra = "allow"
        # allow_arbitrary_types = True

    def initialize_hook(self, conversation: "Conversation") -> Any:
        """Initialize a hook for the plugin."""
        raise NotImplementedError

    def cleanup_hook(self, conversation: "Conversation") -> Any:
        """Cleanup a hook for the plugin."""
        raise NotImplementedError

    def add_message_hook(self, conversation: "Conversation", message: "Message") -> Any:
        """Add a message hook for the plugin."""
        raise NotImplementedError

    def pre_send_hook(self, conversation: "Conversation") -> Any:
        """Pre-send hook for the plugin."""
        raise NotImplementedError

    def post_send_hook(self, conversation: "Conversation", response: "Message") -> Any:
        """Post-send hook for the plugin."""
        raise NotImplementedError


class Message(SMBaseModel):
    """A message in a conversation."""

    role: MESSAGE_ROLE
    text: str
    meta: Dict[str, Any] = {}
    raw: Optional[Any] = None
    llm_model: Optional[str] = None
    llm_provider: Optional[str] = None

    def __str__(self):
        return f"<Message role={self.role} text={self.text!r}>"

    @classmethod
    def from_raw_response(cls, *, text: str, raw: Any) -> "Message":
        """Create a Message instance from a raw response.

        Args:
            text (str): The message text.
            raw (Any): The raw response data.

        Returns:
            Message: A new Message instance.
        """
        self = cls()
        self.text = text
        self.raw = raw
        return self


class Conversation(SMBaseModel):
    """A conversation between a user and an assistant."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    messages: List[Message] = []
    llm_model: Optional[str] = None
    llm_provider: Optional[str] = None
    plugins: List[BasePlugin] = []

    def __str__(self):
        return f"<Conversation id={self.id!r}>"

    def __enter__(self):
        # Execute all initialize hooks.
        for plugin in self.plugins:
            if hasattr(plugin, "initialize_hook"):
                try:
                    plugin.initialize_hook(self)
                except NotImplementedError:
                    pass

        return self

    def __exit__(
        self,
        exc_type: type[BaseException],
        exc_value: BaseException,
        traceback: TracebackType,
    ) -> None:
        """Execute all cleanup hooks."""
        for plugin in self.plugins:
            if hasattr(plugin, "cleanup_hook"):
                try:
                    plugin.cleanup_hook(self)
                except NotImplementedError:
                    pass

    def prepend_system_message(self, text: str, meta: Dict[str, Any] | None = None):
        """Prepend a system message to the conversation."""
        self.messages = [
            Message(role="system", text=text, meta=meta or {})
        ] + self.messages

    def add_message(
        self,
        role: MESSAGE_ROLE = "user",
        text: str | None = None,
        *,
        meta: Optional[Dict[str, Any]] = None,
    ):
        """Add a new message to the conversation."""

        assert text is not None

        # Ensure meta is a dict.
        if meta is None:
            meta = {}

        # Execute all add-message hooks.
        for plugin in self.plugins:
            if hasattr(plugin, "add_message_hook"):
                try:
                    plugin.add_message_hook(
                        self, Message(role=role, text=text, meta=meta)
                    )
                except NotImplementedError:
                    pass

        # Add the message to the conversation.
        self.messages.append(Message(role=role, text=text, meta=meta))

    def send(
        self,
        llm_model: str | None = None,
        llm_provider: str | None = None,
    ) -> Message:
        """Send the conversation to the LLM."""

        # TODO: llm_model and llm_provider should override the conversation's.

        # Execute all pre send hooks.
        for plugin in self.plugins:
            if hasattr(plugin, "pre_send_hook"):
                try:
                    plugin.pre_send_hook(self)
                except NotImplementedError:
                    pass

        # Find the provider and send the conversation.
        provider = find_provider(llm_provider or self.llm_provider)
        response = provider.send_conversation(self)

        # Execute all post-send hooks.
        for plugin in self.plugins:
            if hasattr(plugin, "post_send_hook"):
                try:
                    plugin.post_send_hook(self, response)
                except NotImplementedError:
                    pass

        # Add the response to the conversation.
        self.add_message(role="assistant", text=response.text, meta=response.meta)

        return response

    def get_last_message(self, role: MESSAGE_ROLE) -> Message | None:
        """Get the last message with the given role."""
        return next((m for m in reversed(self.messages) if m.role == role), None)

    def add_plugin(self, plugin: BasePlugin) -> None:
        """Add a plugin to the conversation."""
        self.plugins.append(plugin)
