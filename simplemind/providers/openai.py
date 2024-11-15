from functools import cached_property
from typing import TYPE_CHECKING, Callable, Iterator, Type, TypeVar

import instructor
from pydantic import BaseModel

from ..logging import logger
from ..settings import settings
from ._base import BaseProvider
from ._base_tools import BaseTool

if TYPE_CHECKING:
    from ..models import Conversation, Message

T = TypeVar("T", bound=BaseModel)


class OpenAITool(BaseTool):
    def get_response_schema(self):
        assert self.is_executed, f"Tool {self.name} was not executed."
        assert isinstance(
            self.tool_id, str
        ), f"Expected str for `tool_id` got {self.tool_id!r}"

        return {
            "role": "tool",
            "tool_call_id": self.tool_id,
            "content": self.function_result,
        }

    @logger
    def handle(self, response, messages) -> None:
        """Handle the tool execution result from an API response."""
        tool_used = False

        # Get the message from the response
        assistant_message = response.choices[0].message

        # Check if there's a tool call
        if assistant_message.tool_calls:
            tool_call = assistant_message.tool_calls[
                0
            ]  # Get the first tool call
            if tool_call.function.name == self.name:
                # Execute the function
                import json

                function_args = json.loads(tool_call.function.arguments)
                self.function_result = str(self.raw_func(**function_args))
                self.tool_id = tool_call.id
                tool_used = True

                # Add assistant's message with tool call
                messages.append(
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": tool_call.id,
                                "type": "function",
                                "function": {
                                    "name": tool_call.function.name,
                                    "arguments": tool_call.function.arguments,
                                },
                            }
                        ],
                    }
                )

            if tool_used:
                # Add tool response message
                messages.append(self.get_response_schema())

    def get_input_schema(self):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.get_properties_schema(),
                    "required": self.required,
                    "additionalProperties": False,
                },
            },
        }


class OpenAI(BaseProvider):
    NAME = "openai"
    DEFAULT_MODEL = "gpt-4o-mini"
    DEFAULT_MAX_TOKENS = None
    DEFAULT_KWARGS = {}
    supports_streaming = True

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or settings.get_api_key(self.NAME)

    @cached_property
    def client(self):
        """The raw OpenAI client."""
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        try:
            import openai as oa
        except ImportError as exc:
            raise ImportError(
                "Please install the `openai` package: `pip install openai`"
            ) from exc
        return oa.OpenAI(api_key=self.api_key)

    @cached_property
    def structured_client(self):
        """A OpenAI client with Instructor."""
        return instructor.from_openai(self.client)

    @logger
    def send_conversation(
        self,
        conversation: "Conversation",
        tools: list[Callable | BaseTool] | None = None,
        **kwargs,
    ) -> "Message":
        """Send a conversation to the OpenAI API."""
        from ..models import Message

        # Format messages from conversation
        formatted_messages = [
            {"role": msg.role, "content": msg.text}
            for msg in conversation.messages
        ]

        # Set up tools if provided
        converted_tools = self.make_tools(tools)
        tools_config = (
            [t.get_input_schema() for t in converted_tools] if tools else None
        )

        # Merge all kwargs
        request_kwargs = {
            **self.DEFAULT_KWARGS,
            **kwargs,
            "model": conversation.llm_model or self.DEFAULT_MODEL,
            "messages": formatted_messages,
        }

        if tools_config:
            request_kwargs["tools"] = tools_config

        # Make initial API call
        response = self.client.chat.completions.create(**request_kwargs)

        # Handle tool responses if needed
        while response.choices[0].message.tool_calls:
            print(response)
            # Handle each tool call
            for tool in converted_tools:
                tool.handle(response, formatted_messages)
                if tool.is_executed():
                    # Make another API call with the updated messages
                    response = self.client.chat.completions.create(
                        **request_kwargs
                    )
                    tool.reset_result()

        final_message = response.choices[0].message.content

        return Message(
            role="assistant",
            text=final_message or "",
            raw=response,
            llm_model=conversation.llm_model or self.DEFAULT_MODEL,
            llm_provider=self.NAME,
        )

    @logger
    def structured_response(
        self,
        prompt: str,
        response_model: Type[T],
        *,
        llm_model: str | None = None,
        **kwargs,
    ) -> T:
        """Get a structured response from the OpenAI API."""
        # Ensure messages are provided in kwargs
        messages = [
            {"role": "user", "content": prompt},
        ]
        response = self.structured_client.chat.completions.create(
            messages=messages,
            model=llm_model or self.DEFAULT_MODEL,
            response_model=response_model,
            **{**self.DEFAULT_KWARGS, **kwargs},
        )
        return response_model.model_validate(response)

    @logger
    def generate_text(
        self, prompt: str, *, llm_model: str | None = None, **kwargs
    ):
        """Generate text using the OpenAI API."""
        messages = [
            {"role": "user", "content": prompt},
        ]
        response = self.client.chat.completions.create(
            messages=messages,
            model=llm_model or self.DEFAULT_MODEL,
            **{**self.DEFAULT_KWARGS, **kwargs},
        )
        return response.choices[0].message.content

    @logger
    def generate_stream_text(
        self, prompt: str, *, llm_model: str | None = None, **kwargs
    ) -> Iterator[str]:
        """Generate streaming text using the OpenAI API.

        Yields chunks of text as they are generated by the model.
        """
        messages = [
            {"role": "user", "content": prompt},
        ]
        response = self.client.chat.completions.create(
            messages=messages,
            model=llm_model or self.DEFAULT_MODEL,
            stream=True,  # Enable streaming
            **{**self.DEFAULT_KWARGS, **kwargs},
        )

        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content

    @cached_property
    def tool(self) -> Type[BaseTool]:
        """The tool implementation for OpenAI."""
        return OpenAITool
