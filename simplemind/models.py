from pydantic import BaseModel

class AIRequest(BaseModel):
    prompt: str
    parameters: dict = {}

class AIResponse(BaseModel):
    response: Any
    metadata: dict = {}
