from _context import sm

# Defaults to the default provider (openai)
r = sm.generate_stream_text("Write a poem about the moon", llm_model="gpt-4o-mini")

for chunk in r:
    print(chunk, end="", flush=True)
