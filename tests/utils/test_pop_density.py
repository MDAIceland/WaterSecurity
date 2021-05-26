import sys

sys.path.append("water_security")
from utils.geo import get_average_1k_population_density


def test_tokyo():
    print(
        "Tokyo population density:",
        get_average_1k_population_density(35.6897, 139.6922),
    )
