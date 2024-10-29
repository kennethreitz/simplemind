from _context import sm


class SimpleMemoryPlugin:
    def __init__(self):
        self.memories = [
            "the earth has fictionally beeen destroyed.",
            "the moon is made of cheese.",
        ]

    def yield_memories(self):
        return (m for m in self.memories)

    def send_hook(self, conversation: sm.Conversation):
        for m in self.yield_memories():
            conversation.add_message(role="system", text=m)


conversation = sm.create_conversation(llm_model="grok-beta", llm_provider="xai")
conversation.add_plugin(SimpleMemoryPlugin())

conversation.add_message(
    role="user",
    text="Please write a poem about the moon",
)

r = conversation.send()
print(r.text)
