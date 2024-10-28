import simplemind as sm
from pydantic import BaseModel


class Poem(BaseModel):
    title: str
    content: str


output = sm.structured_response(
    "Write a poem about love",
    llm_model="gpt-4o-mini",
    llm_provider="openai",
    response_model=Poem,
)

print(output)
