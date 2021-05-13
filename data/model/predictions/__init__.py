import pandas as pd
import os
import pickle

PREDICTIONS_DIR = os.sep.join(os.path.split(__file__)[:-1])
FILLED_DATASET_PATH = os.path.join(PREDICTIONS_DIR, "filled_dataset.csv")
PREDICTION_MASK_PATH = os.path.join(PREDICTIONS_DIR, "prediction_mask.csv")
try:
    with open(FILLED_DATASET_PATH, "rb") as inp:
        FILLED_DATASET = pickle.load(inp)
    with open(PREDICTION_MASK_PATH, "rb") as inp:
        PREDICTION_MASK = pickle.load(inp)
except IOError:
    pass
