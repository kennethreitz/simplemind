import openai as oa
import instructor

from ..settings import settings

PROVIDER_NAME = "openai"
DEFAULT_MODEL = "gpt-4o-mini"


class OpenAI:
    __name__ = PROVIDER_NAME
    DEFAULT_MODEL = DEFAULT_MODEL

    def __init__(self, api_key: str = None):
        self.api_key = api_key or settings.OPENAI_API_KEY

    @property
    def client(self):
        """The raw OpenAI client."""
        return oa.OpenAI(api_key=self.api_key)

    @property
    def structured_client(self):
        """A client patched with Instructor."""
        return instructor.patch(oa.OpenAI(api_key=self.api_key))

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
        # Ensure messages are provided in kwargs
        messages = [
            {"role": "user", "content": prompt},
        ]

        response = self.structured_client.chat.completions.create(
            messages=messages, model=llm_model, response_model=response_model
        )
        return response

    def generate_text(self, prompt, *, llm_model):
        messages = [
            {"role": "user", "content": prompt},
        ]

        response = self.client.chat.completions.create(
            messages=messages, model=llm_model
        )

        return response.choices[0].message.content
