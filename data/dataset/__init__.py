import pandas as pd
import os

DATASET_DIR = os.sep.join(os.path.split(__file__)[:-1])
DATASET_PATH = os.path.join(DATASET_DIR, "dataset.csv")
try:
    DATASET = pd.read_csv(DATASET_PATH)
except IOError:
    pass
