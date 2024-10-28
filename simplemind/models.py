from pydantic import BaseModel
from typing import Any, ClassVar


class AIRequest(BaseModel):
    text: str
    parameters: dict = {}

    def __str__(self):
        return self.text


class AIResponse(BaseModel):
    text: str
    response: Any
    metadata: dict = {}

    def __str__(self):
        return self.text
