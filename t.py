import os
from pprint import pprint
from pydantic import BaseModel
import simplemind
from simplemind.concepts import Context
from simplemind.plugins.kv import KVPlugin
from simplemind.plugins.basic_memory import BasicMemoryPlugin
from simplemind.chains.reverse_text import ReverseTextChain
from simplemind.client import Client


class CustomContext(Context):
    def __init__(self):
        super().__init__()
        self.add_plugin("kv", KVPlugin())
        # self.add_plugin("basic_memory", BasicMemoryPlugin())


# Initialize context and client
ctx = CustomContext()
aiclient = Client(
    context=ctx,
    api_key=os.environ["OPENAI_API_KEY"],
)

# Test connection and available models
print(aiclient.available_models)

# Example usage
conversation = aiclient.create_conversation(provider="openai")
response = aiclient.send_message(
    conversation, "Who is Kenneth Reitz?", provider="openai"
)
print(response)

reverse_chain = ReverseTextChain()
result = reverse_chain.run("Hello, World!")
print(result)  # Output: !dlroW ,olleH
