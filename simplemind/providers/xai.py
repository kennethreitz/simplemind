import openai as oa
import instructor

# from ..models import Conversation, Message
from ..settings import settings

PROVIDER_NAME = "xai"
DEFAULT_MODEL = "grok-beta"
BASE_URL = "https://api.x.ai/v1"
DEFAULT_MAX_TOKENS = 1000


class XAI:
    __name__ = PROVIDER_NAME
    DEFAULT_MODEL = DEFAULT_MODEL

    def __init__(self, api_key: str = None):
        self.api_key = api_key or settings.XAI_API_KEY

    @property
    def client(self):
        """The raw OpenAI client."""

        return oa.OpenAI(
            api_key=settings.XAI_API_KEY,
            base_url="https://api.x.ai/v1",
        )

    @property
    def structured_client(self):
        """A client patched with Instructor."""
        return instructor.patch(
            oa.OpenAI(api_key=self.api_key, base_url="https://api.x.ai/v1")
        )

    def send_conversation(self, conversation: "Conversation"):
        """Send a conversation to the OpenAI API."""
        from ..models import Message

        messages = [
            {"role": msg.role, "content": msg.text} for msg in conversation.messages
        ]

        response = self.client.chat.completions.create(
            model=conversation.llm_model or DEFAULT_MODEL, messages=messages
        )

        # Get the response content from the OpenAI response
        assistant_message = response.choices[0].message

        # Create and return a properly formatted Message instance
        return Message(
            role="assistant",
            text=assistant_message.content,
            raw=response,
            llm_model=conversation.llm_model or DEFAULT_MODEL,
            llm_provider=PROVIDER_NAME,
        )

    def structured_response(self, prompt, response_model, *, llm_model):
        raise NotImplementedError("XAI does not support structured responses")

    def generate_text(self, prompt, *, llm_model):
        messages = [
            {"role": "user", "content": prompt},
        ]

        response = self.client.chat.completions.create(
            messages=messages, model=llm_model
        )

        return response.choices[0].message.content
