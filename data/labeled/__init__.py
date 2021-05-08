import pandas as pd
import os

LABELED_DIR = os.sep.join(os.path.split(__file__)[:-1])
CWA_PATH = os.path.join(LABELED_DIR, "2018_-_Cities_Water_Actions.csv")
CWR_PATH = os.path.join(LABELED_DIR, "2018_-_Cities_Water_Risks.csv")
CWA = pd.read_csv(CWA_PATH)
CWR = pd.read_csv(CWR_PATH)

try:
    LABELED_CITIES_PATH = os.path.join(LABELED_DIR, "labeled_cities.csv")
    LABELED_CITIES = pd.read_csv(LABELED_CITIES_PATH)
    RISKS_MAPPING_PATH = os.path.join(LABELED_DIR, "risks_mapping.csv")
    RISKS_MAPPING = (
        pd.read_csv(RISKS_MAPPING_PATH).set_index("code").to_dict()["description"]
    )
except FileNotFoundError:
    pass
