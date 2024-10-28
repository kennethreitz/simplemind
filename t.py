import simplemind

context = None

openai = simplemind.integrations.OpenAI()
print(openai.login())
