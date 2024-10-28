import simplemind

context = None

openai = simplemind.integrations.OpenAI()

print(openai.test_connection())
print(openai.available_models)

print()
print()
message = "who is kennethreitz?"

print(f"> {message}")
print(openai.single_message(message))

# claude = simplemind.integrations.Anthropic()

# # print(claude.test_connection())
# # print(claude.available_models)

# claude.login()
