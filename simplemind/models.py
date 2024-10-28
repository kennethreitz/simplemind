from pydantic import BaseModel
from typing import Any, Dict, List, Optional
import uuid


class AIRequest(BaseModel):
    text: str
    parameters: Dict[str, Any] = {}

    def __str__(self):
        return self.text


class AIResponse(BaseModel):
    text: str
    response: Any
    metadata: Dict[str, Any] = {}

    def __str__(self):
        return self.text


class Message(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class Conversation(BaseModel):
    id: str
    messages: List[Message] = []
    context: Optional[Dict[str, Any]] = {}


class ConversationRequest(BaseModel):
    conversation_id: Optional[str] = None
    message: str
    context_update: Optional[Dict[str, Any]] = None


class ConversationResponse(BaseModel):
    conversation_id: str
    messages: List[Message]
    metadata: Dict[str, Any] = {}
