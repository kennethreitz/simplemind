from simplemind import SimpleMind

sm = SimpleMind()

# The provider will automatically use OPENAI_API_KEY from environment
conversation = sm.create_conversation()
r = sm.add_message(conversation.id, "Who is Kenneth Reitz?")
print(r)
#
