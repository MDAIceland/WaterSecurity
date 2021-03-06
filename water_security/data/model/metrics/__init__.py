import pandas as pd
import os
import pickle

METRICS_DIR = os.sep.join(os.path.split(__file__)[:-1])
VALIDATION_METRICS_PATH = os.path.join(METRICS_DIR, "validation_metrics.pkl")
TRAINING_METRICS_PATH = os.path.join(METRICS_DIR, "training_metrics.pkl")
FEATURES_IMPORTANCES_PATH = os.path.join(METRICS_DIR, "features_importances.pkl")
try:
    with open(VALIDATION_METRICS_PATH, "rb") as inp:
        VALIDATION_METRICS = pickle.load(inp)
    with open(TRAINING_METRICS_PATH, "rb") as inp:
        TRAINING_METRICS = pickle.load(inp)
    with open(FEATURES_IMPORTANCES_PATH, "rb") as inp:
        FEATURES_IMPORTANCES = pickle.load(inp)

except IOError:
    pass
