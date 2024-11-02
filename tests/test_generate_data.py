import pytest

from simplemind.providers import Anthropic, Gemini, OpenAI, Groq, Ollama, Amazon

from pydantic import BaseModel


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
