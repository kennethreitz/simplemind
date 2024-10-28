import simplemind as sm
from pydantic import BaseModel


conversation = sm.create_conversation(llm_model="grok-beta", llm_provider="xai")

conversation.add_message(
    role="user",
    text="Write a poem about love",
)
r = conversation.send()

print(r)
