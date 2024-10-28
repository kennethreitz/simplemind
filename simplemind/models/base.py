from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel

class Message(BaseModel):
    role: str
    content: str
    created_at: datetime = datetime.now()

class Conversation(BaseModel):
    id: str
    messages: List[Message] = []
    context: Dict[str, Any] = {}
    created_at: datetime = datetime.now()
    updated_at: datetime = datetime.now()

    def add_message(self, role: str, content: str) -> Message:
        message = Message(role=role, content=content)
        self.messages.append(message)
        self.updated_at = datetime.now()
        return message

class AIResponse(BaseModel):
    text: str
    response: Any
    metadata: Dict[str, Any] = {} 
