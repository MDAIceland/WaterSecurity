import pandas as pd
import os
import pickle

MODEL_DIR = os.sep.join(os.path.split(__file__)[:-1])
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
try:
    with open(MODEL_PATH, "rb") as inp:
        MODEL = pickle.load(inp)
except IOError:
    pass
