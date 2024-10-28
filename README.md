# SimpleMind: AI for Humans™

**Please Note**: This is a work-in-progress project that needs a lot of work to work properly. Coming soon!

SimpleMind is an AI library designed to simplify your experience with AI APIs in Python. Inspired by a "for humans" philosophy, it abstracts away complexity, giving developers an intuitive and human-friendly way to interact with powerful AI capabilities. With SimpleMind, tapping into AI is as easy as a friendly conversation.

## Features
- **Easy-to-use AI tools**: SimpleMind provides simple interfaces to popular AI services.
- **Human-centered design**: The library prioritizes readability and usability—no need to be an expert to start experimenting.
- **Minimal configuration**: Get started quickly, without worrying about configuration headaches.

## Installation

To install SimpleMind, use pip, eventually—

```bash
pip install simplemind
```



## Quickstart

Here's how easy it is to use SimpleMind to interact with an AI model:

```python
import simplemind as sm
```

SimpleMind takes care of the complex API calls so you can focus on what matters—building, experimenting, and creating.

## Examples

### Text Completion

Generate a response from an AI model based on a given prompt:

```pycon
>>> sm.generate_text(prompt="What is the meaning of life?", provider="anthropic")
```

### Structured Response

```python
class Poem(BaseModel):
    title: str
    content: str
```

```pycon
>>> sm.generate_data(
        "Write a poem about love",
        llm_model="gpt-4o-mini",
        llm_provider="openai",
        response_model=Poem,
    )
title='Eternal Embrace' content='In the quiet hours of the night,\nWhen stars whisper secrets bright,\nTwo hearts beat in a gentle rhyme,\nDancing through the sands of time.\n\nWith every glance, a spark ignites,\nA flame that warms the coldest nights,\nIn laughter shared and whispers sweet,\nLove paints the world, a masterpiece.\n\nThrough stormy skies and sunlit days,\nIn myriad forms, it finds its ways,\nA tender touch, a knowing sigh,\nIn love’s embrace, we learn to fly.\n\nAs seasons change and moments fade,\nIn the tapestry of dreams we’ve laid,\nLove’s threads endure, forever bind,\nA timeless bond, two souls aligned.\n\nSo here’s to love, both bright and true,\nA gift we give, anew, anew,\nIn every heartbeat, every prayer,\nA story written in the air.'
```

### Conversational AI

SimpleMind also allows for easy conversational flows:

```pycon
>>> conversation = sm.create_conversation(llm_model="gpt-4o-mini", llm_provider="openai")

>>> # Add a message to the conversation
>>> conversation.add_message("user", "Hi there, how are you?")

>>> conversation.send()
<Message role=assistant text="Hello! I'm just a computer program, so I don't have feelings, but I'm here and ready to help you. How can I assist you today?">
```

## Supported APIs
- **OpenAI's GPT**
- **Anthropic's Claude**
- **X.AI's Grok**

## Why SimpleMind?
- **Intuitive**: Built with Pythonic simplicity and readability in mind.
- **For Humans**: Emphasizes a human-friendly interface, just like `requests` for HTTP.
- **Open Source**: SimpleMind is open source, and contributions are always welcome!

## Contributing
We welcome contributions of all kinds. Feel free to open issues for bug reports or feature requests, and submit pull requests to make SimpleMind even better.

To get started:

1. Fork the repository.
2. Create a new branch.
3. Make your changes.
4. Submit a pull request.

## Building
1. Clone the repository.
2. `cd` to the root directory.
3. Run `docker-compose up --build`

## License
SimpleMind is licensed under the Apache 2.0 License.

## Acknowledgements
SimpleMind is inspired by the philosophy of "code for humans" and aims to make working with AI models accessible to all. Special thanks to the open-source community for their contributions and inspiration.

---

SimpleMind: Keep it simple, keep it human.

------------------------


## Plugins


SimpleMind supports a plugin system to extend its functionality. Currently available plugins:

- **KVPlugin**: Key-Value storage for context management.
- **BasicMemoryPlugin**: Simple memory storage for conversations.

**Adding a Plugin:**
