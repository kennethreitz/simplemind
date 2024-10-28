from simplemind import SimpleMind

sm = SimpleMind()

# The provider will automatically use OPENAI_API_KEY from environment
conversation = sm.create_conversation()
r = conversation.send_message("Who is Kenneth Reitz?", model="gpt-4o")
print(r)
#
