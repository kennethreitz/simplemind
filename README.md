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
$ pip install simplemind
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
>>> sm.generate_text(prompt="What is the meaning of life?", llm_provider="openai", llm_model="gpt-4o")
"The meaning of life is a profound philosophical question that has been explored by cultures, religions, and philosophers for centuries. Different people and belief systems offer varying interpretations:\n\n1. **Religious Perspectives:** Many religions propose that the meaning of life is to fulfill a divine purpose, serve God, or reach an afterlife. For example, Christianity often emphasizes love, faith, and service to God and others as central to life’s meaning.\n\n2. **Philosophical Views:** Philosophers offer diverse answers. Existentialists like Jean-Paul Sartre argue that life has no inherent meaning, and it is up to individuals to create their own purpose. Others, like Aristotle, suggest that achieving eudaimonia (flourishing or happiness) through virtuous living is the key to a meaningful life.\n\n3. **Scientific and Secular Approaches:** Some people find meaning through understanding the natural world, contributing to human knowledge, or through personal accomplishments and happiness. They may view life's meaning as a product of connection, legacy, or the pursuit of knowledge and creativity.\n\n4. **Personal Perspective:** For many, the meaning of life is deeply personal, involving their relationships, passions, and goals. These individuals define life's purpose through experiences, connections, and the impact they have on others and the world.\n\nUltimately, the meaning of life is a subjective question, with each person finding their own answers based on their beliefs, experiences, and reflections."
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

### Basic Memory Plugin

```python
import simplemind as sm


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
    text="Write a poem about the moon",
)
```
```pycon
>>> conversation.send()
In the vast expanse where stars do play,
There orbits a cheese wheel, far away.
It's not of stone or silver hue,
But cheddar's glow, a sight anew.

In cosmic silence, it does roam,
A lonely traveler, away from home.
No longer does it reflect the sun,
But now it's known for fun begun.

Once Earth's companion, now alone,
A cheese moon orbits, in the dark it's thrown.
Its surface, not of craters wide,
But gouda, swiss, and camembert's pride.

Astronauts of yore, they sought its face,
To find the moon was not a place,
But a haven of dairy delight,
Glowing softly through the night.

In this world, where cheese takes flight,
The moon brings laughter, a whimsical sight.
No longer just a silent sphere,
But a beacon of joy, far and near.

So here's to the moon, in cheese attire,
A playful twist in the cosmic choir.
A reminder that in tales and fun,
The universe is never done.
```


## Supported APIs
- **OpenAI's GPT**
- **Anthropic's Claude**
- **xAI's Grok**

To specify a provider, you can use the `llm_provider` parameter when calling `generate_text`, `generate_data`, or `create_conversation` — you can also pass the `llm_model` parameter to specify a specific LLM provided.

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
3. Generate the requirements.txt file `python -m piptools compile pyproject.toml`
3. Run `docker-compose up --build`

Two containers will run in sequence:
1) `simplemind` - Builds and runs the tests
2) `simplemind-test` - Runs the tests again, if the 1st container suceeds.

## License
SimpleMind is licensed under the Apache 2.0 License.

## Acknowledgements
SimpleMind is inspired by the philosophy of "code for humans" and aims to make working with AI models accessible to all. Special thanks to the open-source community for their contributions and inspiration.

---------------

SimpleMind: Keep it simple, keep it human.
