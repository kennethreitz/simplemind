from _context import sm

conversation = sm.create_conversation(llm_model="gpt-4o", llm_provider="openai")

conversation.add_message(
    "user", "Translate the following text to French: 'Hello, world!'"
)

print(conversation.send().text)
