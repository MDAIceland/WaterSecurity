import os
import pandas as pd

PREPROC_LABELED_DIR = os.sep.join(os.path.split(__file__)[:-1])
LABELED_CITIES_PATH = os.path.join(PREPROC_LABELED_DIR, "labeled_cities.csv")
RISKS_MAPPING_PATH = os.path.join(PREPROC_LABELED_DIR, "risks_mapping.csv")
SEVERITY_MAPPING_PATH = os.path.join(PREPROC_LABELED_DIR, "severity_mapping.csv")
IMPUTATION_REPORT_PATH = os.path.join(
    PREPROC_LABELED_DIR, "labeled_cities_imputation_report.csv"
)
try:
    LABELED_CITIES = pd.read_csv(LABELED_CITIES_PATH)
    RISKS_MAPPING = (
        pd.read_csv(RISKS_MAPPING_PATH).set_index("code").to_dict()["description"]
    )
except FileNotFoundError:
    pass
