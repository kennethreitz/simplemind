import simplemind as sm

res = sm.generate_text(
    prompt="Wish you a happy Diwali!",
    llm_model="gpt-4o",
    llm_provider="openai",
)

print(res)
