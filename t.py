import simplemind

from simplemind.core import settings

print(settings)

# Initialize client without explicit API key
ai = simplemind.Client()

print(ai.available_models)

# The provider will automatically use OPENAI_API_KEY from environment
conversation = ai.create_conversation(provider="openai")
response = ai.send_message(conversation, "Who is Kenneth Reitz?")
print(response)
