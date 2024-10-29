from _context import sm

# Defaults to the default provider (openai)
r = sm.generate_text("Write a poem about the moon", llm_model="gpt-4o-mini")

print(r)
