import simplemind as sm


conversation = sm.create_conversation(llm_model="grok-beta", llm_provider="xai")

conversation.add_message(
    role="user",
    text="Write a poem about working at dominoes",
)
r = conversation.send()

print(r.text)
