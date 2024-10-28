from pprint import pprint
from pydantic import BaseModel
import simplemind

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
