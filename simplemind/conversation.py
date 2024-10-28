class Conversation:
    def __init__(self, ai_client):
        self.messages = []
        self.ai_client = ai_client
    
    def say(self, message):
        self.messages.append({'role': 'user', 'content': message})

    def get_reply(self):
        reply = self.ai_client.message(messages=self.messages)
        self.messages.append({'role': 'system', 'content': reply.text})
        return reply