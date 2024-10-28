# import logging

from pydantic import BaseModel
from typing import Any, Dict, List, Optional
from ..core.models import AIResponse, Conversation, Message
import uuid


DEFAULT_MODEL = "gpt-4o"


class BaseClientProvider:

    def __init__(self, *, model: str = DEFAULT_MODEL, api_key: Optional[str] = None):
        # self.logger = logging.getLogger(self.__class__.__name__)
        self.client = None
        self.model = model
        self._api_key = api_key
        self.conversations: Dict[str, Conversation] = {}

    def login(self):
        """Initializes the AI provider client."""
        msg = "This method must be implemented by the AI provider client."
        raise NotImplementedError(msg)

    def test_connection(self) -> bool:
        """Tests the connection to the AI provider client."""
        msg = "This method must be implemented by the AI provider client."
        raise NotImplementedError(msg)

    def health_check(self):
        """Checks the health of the AI provider client."""
        msg = "This method must be implemented by the AI provider client."
        raise NotImplementedError(msg)

    @property
    def available_models(self) -> List[str]:
        """Returns the available models from the AI provider client."""
        msg = "This method must be implemented by the AI provider client."
        raise NotImplementedError(msg)

    def message(self, message: str, **kwargs) -> AIResponse:
        """Generates a response from the AI provider client."""
        msg = "This method must be implemented by the AI provider client."
        raise NotImplementedError(msg)

    # Uncomment and implement additional methods as needed
    # def features(self):
    #     """Returns the features of the AI provider client."""
    #     msg = "This method must be implemented by the AI provider client."
    #     raise NotImplementedError(msg)

    # def structured_response(self, model, message, **kwargs):
    #     pass

    # def structured_conversation(self, model, message, **kwargs):
    #     pass

    # def single_message(self, model, message, **kwargs):
    #     return self.generate_response(message)

    # def start_conversation(self, model, message, **kwargs):
    #     pass

    def create_conversation(
        self, initial_message: str, context: Optional[Dict[str, Any]] = None
    ) -> Conversation:
        conv_id = str(uuid.uuid4())
        conversation = Conversation(
            id=conv_id,
            messages=[Message(role="user", content=initial_message)],
            context=context or {},
        )
        self.conversations[conv_id] = conversation
        return conversation

    def send_message(
        self,
        conversation_id: str,
        message: str,
        context_update: Optional[Dict[str, Any]] = None,
    ) -> AIResponse:
        if conversation_id not in self.conversations:
            raise ValueError("Conversation ID does not exist.")

        conversation = self.conversations[conversation_id]
        conversation.messages.append(Message(role="user", content=message))

        if context_update:
            conversation.context.update(context_update)

        response = self.generate_response(conversation)
        conversation.messages.append(
            Message(role="assistant", content=response.choices[0].message.content)
        )
        return response

    def generate_response(self, conversation: Conversation) -> AIResponse:
        """Generates a response based on the conversation."""
        raise NotImplementedError(
            "This method must be implemented by the AI provider client."
        )

    def get_conversation(self, conversation_id: str) -> Conversation:
        if conversation_id not in self.conversations:
            raise ValueError("Conversation ID does not exist.")
        return self.conversations[conversation_id]
