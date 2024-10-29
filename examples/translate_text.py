from _context import sm


def translate_to_french(text: str) -> str:
    conversation = sm.create_conversation(llm_model="gpt-4o", llm_provider="openai")

    conversation.add_message(
        "user", f"Translate the following text to French: {text!r}"
    )
    return conversation.send().text


print(translate_to_french("an omlette with cheese"))
