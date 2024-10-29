from _context import sm

from pydantic import BaseModel
import openai
import faiss
import numpy as np
import os
import pickle


class ContextualMemoryPlugin:
    def __init__(self, api_key: str, memory_file: str = "memories.pkl", embedding_model: str = "text-embedding-ada-002"):
        openai.api_key = api_key
        self.memory_file = memory_file
        self.embedding_model = embedding_model
        self.memories = []
        self.embeddings = None
        self.index = None
        self.load_memories()

    def load_memories(self):
        if os.path.exists(self.memory_file):
            with open(self.memory_file, "rb") as f:
                self.memories, self.embeddings = pickle.load(f)
            self.build_faiss_index()
        else:
            self.memories = []
            self.embeddings = []
            self.index = faiss.IndexFlatL2(1536)  # Dimension for ada-002 embeddings

    def save_memories(self):
        with open(self.memory_file, "wb") as f:
            pickle.dump((self.memories, self.embeddings), f)

    def build_faiss_index(self):
        if self.embeddings:
            self.index = faiss.IndexFlatL2(len(self.embeddings[0]))
            self.index.add(np.array(self.embeddings).astype('float32'))
        else:
            self.index = faiss.IndexFlatL2(1536)

    def get_embedding(self, text: str) -> list:
        response = openai.Embedding.create(input=text, model=self.embedding_model)
        return response['data'][0]['embedding']

    def add_memory(self, memory: str):
        embedding = self.get_embedding(memory)
        self.memories.append(memory)
        self.embeddings.append(embedding)
        self.index.add(np.array([embedding]).astype('float32'))
        self.save_memories()

    def retrieve_memories(self, query: str, top_k: int = 3) -> list:
        if not self.index or len(self.embeddings) == 0:
            return []
        query_embedding = self.get_embedding(query)
        D, I = self.index.search(np.array([query_embedding]).astype('float32'), top_k)
        return [self.memories[i] for i in I[0] if i < len(self.memories)]

    def send_hook(self, conversation: sm.Conversation):
        # Retrieve relevant memories based on the latest user message
        if conversation.messages:
            last_user_message = conversation.messages[-1].text
            relevant_memories = self.retrieve_memories(last_user_message)
            for memory in relevant_memories:
                conversation.add_message(role="system", text=memory)

    def on_response(self, conversation: sm.Conversation, response: str):
        # Optionally, add the AI's response to memories
        self.add_memory(response)

# Example Usage

# Define a Pydantic model if needed
class Story(BaseModel):
    title: str
    content: str

# Initialize the conversation with the ContextualMemoryPlugin
memory_plugin = ContextualMemoryPlugin(api_key=sm.settings.OPENAI_API_KEY)

conversation = sm.create_conversation(llm_model="gpt-4o-mini", llm_provider="openai")
conversation.add_plugin(memory_plugin)

# Add user message
conversation.add_message("user", "Tell me a story about a brave knight.")

# Send the conversation and get the response
response = conversation.send()
print(response.text)

# Optionally, retrieve structured data
structured_response = sm.generate_data(
    "Summarize the above story.",
    llm_model="gpt-4o",
    llm_provider="openai",
    response_model=Story,
)
print(structured_response)
