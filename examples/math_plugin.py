from _context import sm


class MathPlugin:
    def send_hook(self, conversation: sm.Conversation):
        last_user_message = conversation.get_last_message(role="user")
        if "calculate" in last_user_message.text.lower():
            expression = last_user_message.text.lower().replace("calculate", "").strip()
            try:
                result = eval(expression)
                conversation.add_message(
                    role="assistant", text=f"The result is {result}."
                )
            except Exception:
                conversation.add_message(
                    role="assistant",
                    text="I'm sorry, I couldn't compute that expression.",
                )


conversation = sm.create_conversation(llm_model="gpt-4o", llm_provider="openai")
conversation.add_plugin(MathPlugin())

conversation.add_message("user", "Calculate 2 + 2 * 3")

print(conversation.send())
