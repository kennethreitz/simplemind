from pydantic import BaseModel
from typing import Any, Dict, List, Optional
from ..concepts.context import Context
from datetime import datetime


class AIRequest(BaseModel):
    text: str
    parameters: Dict[str, Any] = {}

    def __str__(self):
        return self.text


class Message(BaseModel):
    role: str  # "user", "assistant", "system"
    content: str
    created_at: datetime = datetime.now()


class Choice(BaseModel):
    message: Message
    index: int = 0


class AIResponse(BaseModel):
    choices: List[Choice]

    @property
    def content(self) -> str:
        """Helper to get the first message content directly."""
        return self.choices[0].message.content


class Conversation(BaseModel):
    id: str
    messages: List[Message] = []
    context: Optional[Context] = None
    created_at: datetime = datetime.now()
    updated_at: datetime = datetime.now()

    def get_messages(self) -> List[Message]:
        """Returns a list of messages in the conversation."""
        return self.messages

    def add_message(self, role: str, content: str) -> Message:
        """Adds a new message to the conversation."""
        message = Message(role=role, content=content)
        self.messages.append(message)
        self.updated_at = datetime.now()
        return message

    def set_context(self, context: Context):
        """Sets the context for the conversation."""
        self.context = context


class ConversationRequest(BaseModel):
    conversation_id: Optional[str] = None
    message: str
    context_update: Optional[Dict[str, Any]] = None


class ConversationResponse(BaseModel):
    conversation_id: str
    messages: List[Message]
    metadata: Dict[str, Any] = {}
