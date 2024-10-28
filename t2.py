import simplemind as sm

conversation = sm.create_conversation(llm_model="gpt-4o-mini", llm_provider="openai")
conversation.add_message("user", "Hi there, how are you?")
reply = conversation.send()
print(reply)
