from _context import sm

text = sm.generate_text(
    prompt="Write a short summary of 'Pride and Prejudice'.",
    llm_provider="openai",
    llm_model="gpt-4o",
    temperature=0.5,
    max_tokens=150,
)

print(text)
