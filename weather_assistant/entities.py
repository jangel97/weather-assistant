"""City entities for the routing loop's entity key injection."""

from dataclasses import dataclass


@dataclass
class City:
    key: str
    short_name: str
    param_name: str = "city"


CITIES = [
    City(key="new york", short_name="New York"),
    City(key="london", short_name="London"),
    City(key="tokyo", short_name="Tokyo"),
    City(key="paris", short_name="Paris"),
    City(key="berlin", short_name="Berlin"),
    City(key="madrid", short_name="Madrid"),
    City(key="rome", short_name="Rome"),
    City(key="sydney", short_name="Sydney"),
    City(key="toronto", short_name="Toronto"),
    City(key="mumbai", short_name="Mumbai"),
    City(key="beijing", short_name="Beijing"),
    City(key="seoul", short_name="Seoul"),
    City(key="dubai", short_name="Dubai"),
    City(key="singapore", short_name="Singapore"),
    City(key="moscow", short_name="Moscow"),
]
