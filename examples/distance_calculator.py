import math

from _context import sm
from pydantic import Field
from typing_extensions import Literal


@sm.tool(llm_provider="anthropic")
def haversine(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
    unit: Literal["km", "miles"],
) -> float:
    r = 6378.0937 if unit == "km" else 3961
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


def get_user_location() -> str:
    """Get the closest city from the user"""
    return "San Francisco"


def get_coords(
    city_name: str = Field(
        description="The name of the city to take the coordinates from (e.g. London, Rome, Los Angeles)"
    ),
):
    """Get latitude and logitude of a City."""
    _data = {
        "Rome": (41.9028, 12.4964),
        "London": (51.5074, -0.1278),
        "Madrid": (40.4168, -3.7038),
        "San Francisco": (37.7749, -122.4194),
        "Los Angeles": (34.0522, -118.2437),
    }

    return _data.get(city_name)


def distance_calculator(prompt: str):
    conversation = sm.create_conversation(llm_provider="anthropic")
    conversation.add_message("user", prompt)
    return conversation.send(
        tools=[get_user_location, get_coords, haversine]
    ).text


print(distance_calculator("How far is London from where I am?"))
# Prints something like:
"""
The distance between your location (San Francisco) and London is approximately 5,357 miles.
"""

print(
    distance_calculator(
        "What is the distance between Rome and Madrid in Kilometers?"
    )
)


"""
The distance between Rome and Madrid is approximately 1,366 kilometers.
"""
