import pytest
from pydantic import BaseModel

from simplemind.providers import Amazon, Anthropic, Gemini, Groq, Ollama, OpenAI


class ResponseModel(BaseModel):
    result: int


@pytest.mark.parametrize(
    "provider_cls",
    [
        Anthropic,
        Gemini,
        OpenAI,
        Groq,
        Ollama,
        # Amazon
    ],
)
def test_generate_data(provider_cls):
    provider = provider_cls()
    prompt = "What is 2+2?"

    data = provider.structured_response(prompt=prompt, response_model=ResponseModel)

    assert isinstance(data, ResponseModel)
    assert isinstance(data.result, int)
