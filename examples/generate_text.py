from _context import sm

r = sm.generate_text("Write a poem about the moon", llm_provider="openai", llm_model="gpt-3.5-turbo")

print(r)
