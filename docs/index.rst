.. simplemind documentation master file, created by
   sphinx-quickstart on Wed Oct 30 08:08:14 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

SimpleMind: AI for Humans™
==========================

**SimpleMind** is a versatile Python library designed to simplify interactions with various AI models. It provides a consistent and user-friendly interface to numerous AI providers, enabling developers to seamlessly integrate powerful AI capabilities into their applications without the overhead of managing multiple APIs and configurations.

Features
--------

- **Unified Interface**: Interact with multiple AI providers using a single, consistent API
- **Plugin Architecture**: Extend functionality with custom plugins for tasks like memory management and sentiment analysis
- **Structured Data Support**: Generate and manipulate structured data using Pydantic models
- **Human-Centered Design**: Prioritizes readability and ease of use, making AI integration accessible to all developers
- **Minimal Configuration**: Quickly get started without extensive setup or configuration

Supported Providers
------------------

SimpleMind supports a variety of AI providers:

- `OpenAI's GPT <https://openai.com/gpt>`_
- `Anthropic's Claude <https://www.anthropic.com/claude>`_
- `xAI's Grok <https://x.ai/>`_
- `Groq's Groq <https://groq.com/>`_
- `Ollama <https://ollama.com>`_

Installation
-----------

Install SimpleMind using pip:

.. code-block:: shell

    $ pip install simplemind

Quickstart
----------

1. Set your API keys as environment variables:

   .. code-block:: bash

       $ export OPENAI_API_KEY="sk-..."
       $ export ANTHROPIC_API_KEY="..."
       $ export XAI_API_KEY="..."
       $ export GROQ_API_KEY="..."

This is the only required configuration.

2. Import and use SimpleMind:

   .. code-block:: python

       import simplemind as sm

       # Generate text using the default provider (OpenAI)
       response = sm.generate_text("Write a poem about the moon.", llm_model="gpt-4o-mini")
       print(response)

Things to know:

- The primary function for generating text is ``generate_text()``, which is used in the example above.
- To generate structured data, use ``generate_data()``, which most providers support. This is extremely useful.
- The third function, ``create_conversation()``, is used to engage in conversations with AI models.

All of these functions accept an ``llm_model`` and ``llm_provider`` parameter, which allows you to specify the AI model to use. If not provided, the default model for the given provider will be used.


Usage Examples
--------------

Here are some examples demonstrating SimpleMind's key features. From generating creative text and structured data to engaging in conversations and extending functionality with plugins, these examples showcase the library's versatility and ease of use.

Feel free to adapt these examples to your specific use cases!


Text Generation
~~~~~~~~~~~~~~~

This example generates a poem about the moon using the ``gpt-4o-mini`` model.

.. code-block:: python

    import simplemind as sm

    poem = sm.generate_text("Write a poem about the moon.", llm_model="gpt-4o-mini")
    print(poem)

Structured Data Generation
~~~~~~~~~~~~~~~~~~~~~~~~~~

This example generates a poem about love using the ``gpt-4o-mini`` model.

.. code-block:: python

    from pydantic import BaseModel

    class Poem(BaseModel):
        title: str
        content: str

    poem = sm.generate_data(
        prompt="Write a poem about love",
        llm_model="gpt-4o-mini",
        response_model=Poem,
    )
    print(poem)

Conversational AI
~~~~~~~~~~~~~~~~~

This example engages in a conversation with the ``gpt-4o-mini`` model.


.. code-block:: python

    conversation = sm.create_conversation(llm_model="gpt-4o-mini")
    conversation.add_message("user", "Hi there, how are you?")
    response = conversation.send()
    print(response.text)

Plugins
~~~~~~~

This example adds a simple custom memory plugin to the conversation.

.. code-block:: python

    class SimpleMemoryPlugin:
        def __init__(self):
            self.memories = ["the moon is made of cheese."]

        def send_hook(self, conversation):
            for memory in self.memories:
                conversation.add_message(role="system", text=memory)

    conversation = sm.create_conversation()
    conversation.add_plugin(SimpleMemoryPlugin())
    conversation.add_message("user", "Write a poem about the moon")
    print(conversation.send().text)

Plugin Development
~~~~~~~~~~~~~~~~~~

Plugins in SimpleMind follow a simple hook-based architecture. The ``send_hook`` method shown above is just one of several hooks available. Here's a more detailed example showing the complete plugin interface:

.. code-block:: python

    from simplemind.plugins import BasePlugin

    class CustomPlugin(BasePlugin):
        def __init__(self):
            self.conversation_history = []

        def initialize_hook(self, conversation):
            """Called when the plugin is first added to a conversation."""
            print("Plugin initialized!")

        def pre_send_hook(self, conversation):
            """Called before the conversation is sent to the AI provider."""
            # Add any system messages or modify the conversation
            conversation.add_message("system", "Remember to be helpful.")

        def send_hook(self, conversation):
            """Called during the send process."""
            # Add messages or modify the conversation
            self.conversation_history.append(conversation.messages)

        def post_send_hook(self, conversation, response):
            """Called after receiving a response from the AI provider."""
            # Process or modify the response
            return response

        def cleanup_hook(self):
            """Called when the plugin is removed or the conversation ends."""
            self.conversation_history.clear()

All plugins should inherit from ``BasePlugin``, which provides default no-op implementations of these hooks. You only need to implement the hooks you want to use. Here's a simpler example:

.. code-block:: python

    from simplemind.plugins import BasePlugin

    class LoggingPlugin(BasePlugin):
        def pre_send_hook(self, conversation):
            print(f"Sending conversation with {len(conversation.messages)} messages")

        def post_send_hook(self, conversation, response):
            print(f"Received response: {response.text[:50]}...")
            return response

    conversation = sm.create_conversation()
    conversation.add_plugin(LoggingPlugin())
    conversation.add_message("user", "Hello!")
    response = conversation.send()

Plugins can be used to implement features like:

- Conversation logging
- Memory management
- Response filtering
- Token counting
- Custom prompt engineering
- Analytics and monitoring

Multiple plugins can be added to a single conversation, and they will be executed in the order they were added.


Contributing
-----------

1. Fork the Repository
2. Create a New Branch
3. Make Your Changes
4. Submit a Pull Request

Please review our `Code of Conduct <LICENSE>`_ before contributing.

License
-------

SimpleMind is licensed under the `Apache 2.0 License <LICENSE>`_.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   usage
   api
   contributing
   changelog
