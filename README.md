# SimpleMind: AI for Humans™

**Please Note**: This is a work-in-progress project that needs a lot of work to work properly. Coming soon!

SimpleMind is an AI library designed to simplify your experience with AI APIs in Python. Inspired by a "for humans" philosophy, it abstracts away complexity, giving developers an intuitive and human-friendly way to interact with powerful AI capabilities. With SimpleMind, tapping into AI is as easy as a friendly conversation.

## Features
- **Easy-to-use AI tools**: SimpleMind provides simple interfaces to popular AI services.
- **Human-centered design**: The library prioritizes readability and usability—no need to be an expert to start experimenting.
- **Minimal configuration**: Get started quickly, without worrying about configuration headaches.

## Installation

To install SimpleMind, use pip:

```bash
pip install simplemind
```

## Quickstart

Here's how easy it is to use SimpleMind to interact with an AI model:

```python
import simplemind

# Initialize a client
aiclient = simplemind.Client(api_key="YOUR_API_KEY")

# Generate text
generated_text = aiclient.generate_text(prompt="Once upon a time in a land far away...")
print(generated_text)
```

SimpleMind takes care of the complex API calls so you can focus on what matters—building, experimenting, and creating.

## Examples

### Text Completion

Generate a response from an AI model based on a given prompt:

```python
response = aiclient.generate_text(prompt="What is the meaning of life?")
print(response)
```

### Conversational AI

SimpleMind also allows for easy conversational flows:

```python
conversation = aiclient.start_conversation()

# Add a message to the conversation
conversation.say("Hi there, how are you?")

# Get the AI's response
reply = conversation.get_reply()
print(reply)
```

## Supported APIs
- **OpenAI GPT**
- **Cohere**
- **Hugging Face Transformers**

More integrations coming soon!

## Configuration
To use SimpleMind, you'll need an API key from the supported AI provider. Just pass it when initializing the `Client`:

```python
aiclient = simplemind.Client(api_key="YOUR_API_KEY")
```

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
SimpleMind is licensed under the MIT License.

## Community
Join our community to share ideas, get help, or just hang out:

- [GitHub Discussions](https://github.com/simplemind-ai/simplemind/discussions)
- [Discord](https://discord.gg/simplemind)

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
