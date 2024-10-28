from .models import Conversation


class SimpleMind:
    def create_conversation(self, *, llm_model=None, llm_provider=None):
        return Conversation()

    def structured_response(
        self, *, llm_model=None, llm_provider=None, response_model=None
    ):
        pass


def create_conversation():
    return SimpleMind().create_conversation()


globals().update(locals())
