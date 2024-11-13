from functools import cached_property
from typing import TYPE_CHECKING, Any, Callable, Iterator, Type, TypeVar

import instructor
from pydantic import BaseModel

from ..logging import logger
from ..settings import settings
from ._base import BaseProvider
from ._base_tools import BaseTool

if TYPE_CHECKING:
    from ..models import Conversation, Message

T = TypeVar("T", bound=BaseModel)


class AnthropicTool(BaseTool):
    def get_response_schema(self) -> Any:
        assert self.is_executed, f"Tool {self.name} was not executed."
        assert isinstance(
            self.tool_id, str
        ), f"Expected str for `tool_id` got {self.tool_id!r}"
        return {
            "type": "tool_result",
            "tool_use_id": self.tool_id,
            "content": self.function_result,
        }

    @logger
    def handle(self, response, messages) -> None:
        """Handle the tool execution result from an API response."""
        msg = {"role": "assistant", "content": []}
        tool_used = False
        for content in response.content:
            if content.type == "tool_use" and content.name == self.name:
                msg["content"].append(
                    {
                        "type": "tool_use",
                        "id": content.id,
                        "name": content.name,
                        "input": content.input,
                    }
                )
                # Function execution:
                self.function_result = str(self.raw_func(**content.input))
                self.tool_id = content.id
                tool_used = True
            elif content.type == "text":
                msg["content"].append({"type": "text", "text": content.text})

        if tool_used:
            messages.append(msg)
            messages.append(
                {"role": "user", "content": [self.get_response_schema()]}
            )

    def get_input_schema(self):
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": self.get_properties_schema(),
                "required": self.required,
            },
        }


class Anthropic(BaseProvider):
    NAME = "anthropic"
    DEFAULT_MODEL = "claude-3-5-sonnet-20241022"
    DEFAULT_MAX_TOKENS = 1_000
    DEFAULT_KWARGS = {"max_tokens": DEFAULT_MAX_TOKENS}
    supports_streaming = True

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or settings.get_api_key(self.NAME)

    @cached_property
    def client(self):
        """The raw Anthropic client."""
        if not self.api_key:
            raise ValueError("Anthropic API key is required")
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError(
                "Please install the `anthropic` package: `pip install anthropic`"
            ) from exc

        return anthropic.Anthropic(api_key=self.api_key)

    @cached_property
    def structured_client(self):
        """A client patched with Instructor."""
        return instructor.from_anthropic(self.client)

    @logger
    def send_conversation(
        self,
        conversation: "Conversation",
        tools: list[Callable | BaseTool] | None = None,
        **kwargs,
    ) -> "Message":
        """Send a conversation to the Anthropic API."""
        from ..models import Message

        # Format messages from conversation
        formatted_messages = [
            {"role": msg.role, "content": msg.text}
            for msg in conversation.messages
        ]

        # Set up tools if provided
        converted_tools = self.make_tools(tools)
        tools_config = (
            {"tools": [t.get_input_schema() for t in converted_tools]}
            if tools is not None
            else {}
        )

        # Merge all kwargs
        request_kwargs = {
            **self.DEFAULT_KWARGS,
            **kwargs,
            **tools_config,
            "model": conversation.llm_model or self.DEFAULT_MODEL,
            "messages": formatted_messages,
        }

        # Make initial API call
        response = self.client.messages.create(**request_kwargs)

        # Handle tool responses if needed
        while response.content[-1].type != "text":
            # Continue handling tools if the LLM is doing
            # multiple sub-seqequent/sequential tool calls
            for tool in converted_tools:
                tool.handle(response, formatted_messages)
                if tool.is_executed():
                    response = self.client.messages.create(**request_kwargs)
                    # Resetting the tool results in case this tool gets used again
                    tool.reset_result()

        final_message = response.content[-1].text

        return Message(
            role="assistant",
            text=final_message,
            raw=response,
            llm_model=conversation.llm_model or self.DEFAULT_MODEL,
            llm_provider=self.NAME,
        )

    @logger
    def structured_response(
        self, response_model: Type[T], *, llm_model: str | None = None, **kwargs
    ) -> T:
        model = llm_model or self.DEFAULT_MODEL

        # Extract the prompt from kwargs if it exists
        prompt = kwargs.pop("prompt", kwargs.pop("messages", ""))

        # Format the messages properly
        messages = [{"role": "user", "content": prompt}]

        response = self.structured_client.messages.create(
            model=model,
            messages=messages,  # Add the messages parameter
            response_model=response_model,
            **{**self.DEFAULT_KWARGS, **kwargs},
        )
        return response_model.model_validate(response)

    @logger
    def generate_text(self, prompt: str, *, llm_model: str, **kwargs):
        messages = [
            {"role": "user", "content": prompt},
        ]

        response = self.client.messages.create(
            model=llm_model or self.DEFAULT_MODEL,
            messages=messages,
            **{**self.DEFAULT_KWARGS, **kwargs},
        )

        return response.content[0].text

    @logger
    def generate_stream_text(
        self, prompt: str, *, llm_model: str, **kwargs
    ) -> Iterator[str]:
        # Prepare the messages.
        messages = [
            {"role": "user", "content": prompt},
        ]

        # Make the request.
        with self.client.messages.stream(
            model=llm_model or self.DEFAULT_MODEL,
            messages=messages,
            **{**self.DEFAULT_KWARGS, **kwargs},
        ) as stream:
            # Yield each chunk of text from the stream.
            for chunk in stream.text_stream:
                yield chunk

    @cached_property
    def tool(self) -> Type[BaseTool]:
        """The tool implementation for Antrhopic."""
        return AnthropicTool
