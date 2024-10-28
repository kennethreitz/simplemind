from pprint import pprint
from pydantic import BaseModel
import simplemind
from simplemind.vector_store.faiss_store import FAISSStore
import numpy as np

context = None

openai = simplemind.integrations.OpenAI()


class YearlyData(BaseModel):
    year: int
    events: list[str]


class ProjectData(BaseModel):
    name: str
    description: str
    url: str
    github_url: str


class BioData(BaseModel):
    bio: str
    spouse_name: str
    history: list[YearlyData]
    fun_facts: list[str]
    # age: int
    # occupation: str
    # bio: str
    # affiliations: list[str]


class PersonData(BaseModel):
    bio: BioData
    projects: list[ProjectData]
    yearly_breakdown: list[YearlyData]


print(openai.test_connection())
print(openai.available_models)

print()
print()
message = "who is kenneth reitz?"

print(f"> {message}")
pprint(openai.message(message, response_model=BioData))

# claude = simplemind.integrations.Anthropic()

# # print(claude.test_connection())
# # print(claude.available_models)

# claude.login()

vector_store = FAISSStore(dimension=768)  # Example dimension for embeddings

# Add embeddings
embeddings = np.random.random((10, 768)).astype('float32')
ids = [f"doc_{i}" for i in range(10)]
vector_store.add_embeddings(embeddings, ids)

# Search
query_embedding = np.random.random((1, 768)).astype('float32')
results = vector_store.search(query_embedding, top_k=3)
print(results)
