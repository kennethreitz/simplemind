from typing import Annotated, Literal

import pytest
from pydantic import Field

import simplemind as sm

from simplemind.providers import Anthropic, OpenAI
from simplemind.providers._base_tools import BaseTool

MODELS = [
    Anthropic,
    # Gemini,
    OpenAI,
    # Groq,
    # Ollama,
    # Amazon
]


def get_weather(
    location: Annotated[
        str, Field(description="The city and state, e.g. San Francisco, CA")
    ],
    unit: Annotated[
        Literal["celcius", "fahrenheit"],
        Field(description="The unit of temperature, either 'celsius' or 'fahrenheit'"),
    ] = "celcius",
):
    """
    Get the current weather in a given location
    """
    return f"42 {unit}"


def get_location():
    """Get the current location"""
    return "San Francisco,CA"


@pytest.mark.parametrize(
    "provider_cls",
    MODELS,
)
def test_single_tool_args(provider_cls):
    conv = sm.create_conversation(
        llm_model=provider_cls.DEFAULT_MODEL, llm_provider=provider_cls.NAME
    )

    conv.add_message(text="What is the weather in San Francisco?")
    data = conv.send(tools=[get_weather])
    assert "42" in data.text


@pytest.mark.parametrize(
    "provider_cls",
    MODELS,
)
def test_single_tool_no_args(provider_cls):
    conv = sm.create_conversation(
        llm_model=provider_cls.DEFAULT_MODEL, llm_provider=provider_cls.NAME
    )

    conv.add_message(text="What is my current location")
    data = conv.send(tools=[get_location])
    assert "San Francisco" in data.text


@pytest.mark.parametrize(
    "provider_cls",
    MODELS,
)
def test_single_tool_partial(provider_cls):
    conv = sm.create_conversation(
        llm_model=provider_cls.DEFAULT_MODEL, llm_provider=provider_cls.NAME
    )

    conv.add_message(text="Can you tell me the weather?")
    conv.send(tools=[get_weather])
    # Will answer something like:
    """
    I can help you check the weather, but I need to know the location you're interested in.
    Could you please provide a city and state (e.g., "Los Angeles, CA" or "New York, NY")
    where you'd like to know the weather?
    """

    conv.add_message(text="San Francisco, CA")
    data = conv.send(tools=[get_weather])
    assert "42" in data.text


@pytest.mark.parametrize(
    "provider_cls",
    MODELS,
)
def test_multiple_tools(provider_cls):
    conv = sm.create_conversation(
        llm_model=provider_cls.DEFAULT_MODEL, llm_provider=provider_cls.NAME
    )

    conv.add_message(text="What is the wheather at my current location?")
    data = conv.send(tools=[get_location, get_weather])
    assert "San Francisco" in data.text
    assert "42" in data.text


@pytest.mark.parametrize(
    "provider_cls",
    MODELS,
)
def test_tool_decorator(provider_cls):
    @sm.tool(llm_provider=provider_cls.NAME)
    def exchange_rate(currency_pair: str) -> float:
        return 7.9

    assert isinstance(exchange_rate, BaseTool)
    assert exchange_rate.name == "exchange_rate"
    assert list(exchange_rate.properties.keys()) == ["currency_pair"]
