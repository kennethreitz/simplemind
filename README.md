# Simplemind: AI for Humans™

**Keep it simple, keep it human.**

Simplemind is AI library designed to simplify your experience with AI APIs in Python. Inspired by a "for humans" philosophy, it abstracts away complexity, giving developers an intuitive and human-friendly way to interact with powerful AI capabilities.

![simplemind](https://github.com/user-attachments/assets/36df2103-2583-4958-ad5e-19cda7740256)

## Features

With Simplemind, tapping into AI is as easy as a friendly conversation.

- **Easy-to-use AI tools**: Simplemind provides simple interfaces to most popular AI services.
- **Human-centered design**: The library prioritizes readability and usability—no need to be an expert to start experimenting.
- **Minimal configuration**: Get started quickly, without worrying about configuration headaches.

## Supported APIs

The APIs remain identical between all supported providers / models:

<table>
  <thead>
    <tr>
      <th></th>
      <th><code>llm_provider</code></th>
      <th>Default <code>llm_model</code></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><a href="https://www.anthropic.com/claude">Anthropic's Claude</a></td>
      <td><code>"anthropic"</code></td>
      <td><code>"claude-3-5-sonnet-20241022"</code></td>
    </tr>
    <tr>
      <td><a href="https://aws.amazon.com/bedrock/">Amazon's Bedrock</a></td>
      <td><code>"amazon"</code></td>
      <td><code>"anthropic.claude-3-5-sonnet-20241022-v2:0"</code></td>
    </tr>
    <tr>
      <td><a href="https://www.deepseek.com">Deepseek</a></td>
      <td><code>"deepseek"</code></td>
      <td><code>"deepseek-chat"</code></td>
    </tr>
    <tr>
      <td><a href="https://gemini.google/">Google's Gemini</a></td>
      <td><code>"gemini"</code></td>
      <td><code>"models/gemini-1.5-pro"</code></td>
    </tr>
    <tr>
      <td><a href="https://groq.com/">Groq's Groq</a></td>
      <td><code>"groq"</code></td>
      <td><code>"llama3-8b-8192"</code></td>
    </tr>
    <tr>
      <td><a href="https://ollama.com">Ollama</a></td>
      <td><code>"ollama"</code></td>
      <td><code>"llama3.2"</code></td>
    </tr>
    <tr>
      <td><a href="https://openai.com/gpt">OpenAI's GPT</a></td>
      <td><code>"openai"</code></td>
      <td><code>"gpt-4o-mini"</code></td>
    </tr>
    <tr>
      <td><a href="https://x.ai/">xAI's Grok</a></td>
      <td><code>"xai"</code></td>
      <td><code>"grok-beta"</code></td>
    </tr>
  </tbody>
</table>

To specify a specific provider or model, you can use the `llm_provider` and `llm_model` parameters when calling: `generate_text`, `generate_data`, or `create_conversation`.

If you want to see Simplemind support additional providers or models, please send a pull request!

## Quickstart

Simplemind takes care of the complex API calls so you can focus on what matters—building, experimenting, and creating.

```bash
$ pip install 'simplemind[full]'
```

First, authenticate your API keys by setting them in the environment variables:

```bash
$ export OPENAI_API_KEY="sk-..."
```

This pattern allows you to keep your API keys private and out of your codebase. Other supported environment variables: `ANTHROPIC_API_KEY`, `XAI_API_KEY`, `GROQ_API_KEY`, and `GEMINI_API_KEY`.

Next, import Simplemind and start using it:

```python
import simplemind as sm
```

## Examples

Here are some examples of how to use Simplemind.

**Please note**: Most of the calls seen here optionally accept `llm_provider` and `llm_model` parameters, which you provide as strings.

### Text Completion

Generate a response from an AI model based on a given prompt:

```pycon
>>> sm.generate_text(prompt="What is the meaning of life?")
"The meaning of life is a profound philosophical question that has been explored by cultures, religions, and philosophers for centuries. Different people and belief systems offer varying interpretations:\n\n1. **Religious Perspectives:** Many religions propose that the meaning of life is to fulfill a divine purpose, serve God, or reach an afterlife. For example, Christianity often emphasizes love, faith, and service to God and others as central to life’s meaning.\n\n2. **Philosophical Views:** Philosophers offer diverse answers. Existentialists like Jean-Paul Sartre argue that life has no inherent meaning, and it is up to individuals to create their own purpose. Others, like Aristotle, suggest that achieving eudaimonia (flourishing or happiness) through virtuous living is the key to a meaningful life.\n\n3. **Scientific and Secular Approaches:** Some people find meaning through understanding the natural world, contributing to human knowledge, or through personal accomplishments and happiness. They may view life's meaning as a product of connection, legacy, or the pursuit of knowledge and creativity.\n\n4. **Personal Perspective:** For many, the meaning of life is deeply personal, involving their relationships, passions, and goals. These individuals define life's purpose through experiences, connections, and the impact they have on others and the world.\n\nUltimately, the meaning of life is a subjective question, with each person finding their own answers based on their beliefs, experiences, and reflections."
```

### Streaming Text

```python
>>> for chunk in sm.generate_text("Write a poem about the moon", stream=True):
...     print(chunk, end="", flush=True)
```

### Structured Data with Pydantic

You can use Pydantic models to structure the response from the LLM, if the LLM supports it.

```python
class Poem(BaseModel):
    title: str
    content: str
```

```pycon
>>> sm.generate_data("Write a poem about love", response_model=Poem)
title='Eternal Embrace' content='In the quiet hours of the night,\nWhen stars whisper secrets bright,\nTwo hearts beat in a gentle rhyme,\nDancing through the sands of time.\n\nWith every glance, a spark ignites,\nA flame that warms the coldest nights,\nIn laughter shared and whispers sweet,\nLove paints the world, a masterpiece.\n\nThrough stormy skies and sunlit days,\nIn myriad forms, it finds its ways,\nA tender touch, a knowing sigh,\nIn love’s embrace, we learn to fly.\n\nAs seasons change and moments fade,\nIn the tapestry of dreams we’ve laid,\nLove’s threads endure, forever bind,\nA timeless bond, two souls aligned.\n\nSo here’s to love, both bright and true,\nA gift we give, anew, anew,\nIn every heartbeat, every prayer,\nA story written in the air.'
```

#### A more complex example

```python
class InstructionStep(BaseModel):
    step_number: int
    instruction: str

class RecipeIngredient(BaseModel):
    name: str
    quantity: float
    unit: str

class Recipe(BaseModel):
    name: str
    ingredients: list[RecipeIngredient]
    instructions: list[InstructionStep]

recipe = sm.generate_data(
    "Write a recipe for chocolate chip cookies",
    response_model=Recipe,
)
```

Special thanks to [@jxnl](https://github.com/jxnl) for building [Instructor](https://github.com/jxnl/instructor), which makes this possible!

### Conversational AI

SimpleMind also allows for easy conversational flows:

```pycon
>>> conv = sm.create_conversation()

>>> # Add a message to the conversation
>>> conv.add_message("user", "Hi there, how are you?")

>>> conv.send()
<Message role=assistant text="Hello! I'm just a computer program, so I don't have feelings, but I'm here and ready to help you. How can I assist you today?">
```

To continue the conversation, you can call `conv.send()` again, which returns the next message in the conversation:

```pycon
>>> conv.add_message("user", "What is the meaning of life?")
>>> conv.send()
<Message role=assistant text="The meaning of life is a profound philosophical question that has been explored by cultures, religions, and philosophers for centuries. Different people and belief systems offer varying interpretations:\n\n1. **Religious Perspectives:** Many religions propose that the meaning of life is to fulfill a divine purpose, serve God, or reach an afterlife. For example, Christianity often emphasizes love, faith, and service to God and others as central to life’s meaning.\n\n2. **Philosophical Views:** Philosophers offer diverse answers. Existentialists like Jean-Paul Sartre argue that life has no inherent meaning, and it is up to individuals to create their own purpose. Others, like Aristotle, suggest that achieving eudaimonia (flourishing or happiness) through virtuous living is the key to a meaningful life.\n\n3. **Scientific and Secular Approaches:** Some people find meaning through understanding the natural world, contributing to human knowledge, or through personal accomplishments and happiness. They may view life’s meaning as a product of connection, legacy, or the pursuit of knowledge and creativity.\n\n4. **Personal Perspective:** For many, the meaning of life is deeply personal, involving their relationships, passions, and goals. These individuals define life’s purpose through experiences, connections, and the impact they have on others and the world.\n\nUltimately, the meaning of life is a subjective question, with each person finding their own answers based on their beliefs, experiences, and reflections.">
```

### Stop Repeating Yourself

You can use the `Session` class to set default parameters for all calls:

```python
# Create a session with defaults
gpt_4o_mini = sm.Session(llm_provider="openai", llm_model="gpt-4o-mini")

# Now all calls use these defaults
response = gpt_4o_mini.generate_text("Hello!")
conversation = gpt_4o_mini.create_conversation()
```

This maintains the simplicity of the original API while reducing repetition.

The session object also supports overriding defaults on a per-call basis:

```python
response = gpt_4o_mini.generate_text("Complex task here", llm_model="gpt-4")
```

### Basic Memory Plugin

Harnessing the power of Python, you can easily create your own plugins to add additional functionality to your conversations:

```python
class SimpleMemoryPlugin(sm.BasePlugin):
    def __init__(self):
        self.memories = [
            "the earth has fictionally beeen destroyed.",
            "the moon is made of cheese.",
        ]

    def yield_memories(self):
        return (m for m in self.memories)

    def pre_send_hook(self, conversation: sm.Conversation):
        for m in self.yield_memories():
            conversation.add_message(role="system", text=m)


conversation = sm.create_conversation()
conversation.add_plugin(SimpleMemoryPlugin())


conversation.add_message(
    role="user",
    text="Please write a poem about the moon",
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

Simple, yet effective.

### Tools (Function calling)
Tools (also known as functions) let you call any Python function from your AI conversations. Here's an example:

```python
def get_weather(
    location: Annotated[
        str, Field(description="The city and state, e.g. San Francisco, CA")
    ],
    unit: Annotated[
        Literal["celcius", "fahrenheit"],
        Field(
            description="The unit of temperature, either 'celsius' or 'fahrenheit'"
        ),
    ] = "celcius",
):
    """
    Get the current weather in a given location
    """
    return f"42 {unit}"

# Add your function as a tool
conversation = sm.create_conversation()
conversation.add_message("user", "What's the weather in San Francisco?")
response = conversation.send(tools=[get_weather])
```

Note how we're using Python's `Annotated` feature combined with `Field` to provide additional context to our function parameters. This helps the AI understand the intention and constraints of each parameter, making tool calls more accurate and reliable.
You can alos ommit `Annotated` and just pass the `Field` parameter.
```python
def get_weather(
    location: str = Field(description="The city and state, e.g. San Francisco, CA"),
    unit:Literal["celcius", "fahrenheit"]= Field(
        default="celcius",
        description="The unit of temperature, either 'celsius' or 'fahrenheit'"
        ),
):
    """
    Get the current weather in a given location
    """
    return f"42 {unit}"
```

Functions can be defined with type hints and Pydantic models for validation. The LLM will intelligently choose when to call the functions and incorporate the results into its responses.

#### 🪄 Using LLM for automatic tool definition (Experimental)

Simplemind provides a decorator to automatically transform Python functions into tools with AI-generated metadata. Simply use the `@simplemind.tool` decorator to have the LLM analyze your function and generate appropriate descriptions and schema:

```python
@simplemind.tool(llm_provider="anthropic")
def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = r * c
    return d
```
Notice how we have not added any docstrings or `Field` for the function.
The decorator will use the specified LLM provider to generate the tool schema, including descriptions and parameter details:

```json
{
    "name": "haversine",
    "description": "Calculates the great-circle distance between two points on Earth given their latitude and longitude coordinates",
    "input_schema": {
        "type": "object",
        "properties": {
            "lat1": {
                "type": "number",
                "description": "Latitude of the first point in decimal degrees",
            },
            "lon1": {
                "type": "number",
                "description": "Longitude of the first point in decimal degrees",
            },
            "lat2": {
                "type": "number",
                "description": "Latitude of the second point in decimal degrees",
            },
            "lon2": {
                "type": "number",
                "description": "Longitude of the second point in decimal degrees",
            }
        },
        "required": ["lat1", "lon1", "lat2", "lon2"],
    },
}
```

The decorated function can then be used like any other tool with the conversation API.

```python
conversation = sm.create_conversation()
conversation.add_message("user", "How far is London from my location")
response = conversation.send(tools=[get_location, get_coords, haversine]) # Multiple tools can be passed
```

See [examples/distance_calculator.py](examples/distance_calculator.py) for more.

### Logging

Simplemind uses [Logfire](https://pydantic.dev/logfire) for logging. To enable logging, call `sm.enable_logfire()`.

### More Examples

Please see the [examples](examples) directory for executable examples.

---

## Contributing

We welcome contributions of all kinds. Feel free to open issues for bug reports or feature requests, and submit pull requests to make SimpleMind even better.

To get started:

1. Fork the repository.
2. Create a new branch.
3. Make your changes.
4. Submit a pull request.

## License

Simplemind is licensed under the Apache 2.0 License.

## Acknowledgements

Simplemind is inspired by the philosophy of "code for humans" and aims to make working with AI models accessible to all. Special thanks to the open-source community for their contributions and inspiration.
