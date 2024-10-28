from .models import Conversation
from .utils import find_provider


class SimpleMind:

    def structured_response(
        self, prompt, *, llm_model=None, llm_provider=None, response_model=None
    ):
        provider = find_provider(llm_provider)

        return provider.structured_response(
            llm_model=llm_model, response_model=response_model, prompt=prompt
        )


def create_conversation(llm_model=None, llm_provider=None):
    return Conversation(llm_model=llm_model, llm_provider=llm_provider)


def structured_response(
    prompt, *, llm_model=None, llm_provider=None, response_model=None
):
    provider = find_provider(llm_provider)

    return provider.structured_response(
        prompt=prompt,
        llm_model=llm_model,
        response_model=response_model,
    )


globals().update(locals())
