import simplemind

aiclient = simplemind.Ollama()

print("Messaging client")
message_response = aiclient.message(message="Once upon a time in a land far away...")
print(message_response)

print("Generating Text")
generated_text = aiclient.generate_text(prompt="Once upon a time in a land far away...")
print(generated_text)

print("Initiating Conversation")
conversation = aiclient.start_conversation()

# Add a message to the conversation
conversation.say("Please remember the number 42")

# Get the AI's response
reply = conversation.get_reply()
print(reply)

# Add another message to the conversation
conversation.say("What number did I ask you to remember?")

# Get the AI's response
reply = conversation.get_reply()
print(reply)
