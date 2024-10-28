import simplemind as sm

conversation = sm.create_conversation()
conversation.add_message(role="user", text="Hello, how are you? Do you like poetry?")
r = conversation.send(llm_provider="openai")

print(r.text)
print(conversation.messages)
