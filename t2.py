import instructor
from pydantic import BaseModel
from openai import OpenAI


class ProjectInfo(BaseModel):
    name: str
    description: str
    url: str
    github_url: str


# Define your desired output structure
class UserInfo(BaseModel):
    name: str
    age: int
    bio: str
    projects: list[ProjectInfo]


# Patch the OpenAI client
client = instructor.from_openai(OpenAI())

# Extract structured data from natural language
user_info = client.chat.completions.create(
    model="gpt-4o",
    response_model=UserInfo,
    messages=[{"role": "user", "content": "who is kennethreitz?"}],
)

print(user_info.model_dump())
# > 30
