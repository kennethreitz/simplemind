import simplemind as sm

conversation = sm.create_conversation()
conversation.add_message(role="user", text="Hello, how are you?")
r = conversation.send(llm_model="gpt-4o-mini", llm_provider="openai")

print(r)
