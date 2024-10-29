import unittest
import simplemind as sm
from pydantic import BaseModel

class TestOllama(unittest.TestCase):

  def test_generate_text(self):
    result = sm.generate_text(prompt="What is the meaning of life?", llm_provider="ollama", llm_model="llama3.2")
    self.assertIsNotNone(result)

  def test_create_conversation(self):
    conversation = sm.create_conversation(llm_provider="ollama", llm_model="llama3.2")
    conversation.add_message("user", "Remember the number 42.")
    result = conversation.send()
    self.assertIsNotNone(result)
    self.assertIsInstance(result, sm.models.Message)

  def test_memory(self):
    conversation = sm.create_conversation(llm_provider="ollama", llm_model="llama3.2")
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
    
    conversation.add_plugin(SimpleMemoryPlugin())
    conversation.add_message(
        role="user",
        text="Write a poem about the moon",
    )
    result = conversation.send()
    self.assertIsNotNone(result)
    self.assertIsInstance(result, sm.models.Message)

  def test_structure_response(self):
    class Poem(BaseModel):
        title: str
        content: str
    with self.assertRaises(NotImplementedError):
      data_obj = sm.generate_data(
        prompt="Write a poem about love",
        llm_provider="ollama",
        llm_model="llama3.2",
        response_model=Poem)
      self.assertIsNotNone(data_obj)
      self.assertIsInstance(data_obj, Poem)


if __name__ == '__main__':
  unittest.main()