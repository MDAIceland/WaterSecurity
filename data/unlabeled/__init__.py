import os
import pandas as pd

UNLABELED_DIR = os.sep.join(os.path.split(__file__)[:-1])
WORLD_CITIES_PATH = os.path.join(UNLABELED_DIR, "worldcities.csv")
WORLD_CITIES = pd.read_csv(WORLD_CITIES_PATH)
BIG_CITIES_PATH = os.path.join(UNLABELED_DIR, "bigcities.csv")
BIG_CITIES_ALL_COUNTRIES_PATH = os.path.join(
    UNLABELED_DIR, "bigcities_allcountries.csv"
)
try:
    BIG_CITIES = pd.read_csv(BIG_CITIES_PATH)
    BIG_CITIES_ALL_COUNTRIES = pd.read_csv(BIG_CITIES_ALL_COUNTRIES_PATH)
except FileNotFoundError:
    pass
try:
    COUNTRIES_DATASET = pd.read_csv(
        os.path.join(UNLABELED_DIR, "preprocessed_countries_dataset.csv"), index_col=0
    )
except ImportError:
    pass
