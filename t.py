import simplemind

context = None

openai = simplemind.integrations.OpenAI()
openai.login()

print(openai.available_models)
