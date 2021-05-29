import pandas as pd
import os
import pickle

MODEL_DIR = os.sep.join(os.path.split(__file__)[:-1])
MODEL_BEST_PARAMS_PATH = os.path.join(MODEL_DIR, "model_best_params.csv")
try:
    with open(MODEL_BEST_PARAMS_PATH, "rb") as inp:
        MODEL_BEST_PARAMS = pd.read_csv(MODEL_BEST_PARAMS_PATH, index_col=0).to_dict()
        for risk, risk_dict in MODEL_BEST_PARAMS.items():
            for param, val in risk_dict.items():
                try:
                    try:
                        risk_dict[param] = int(val)
                    except ValueError:
                        risk_dict[param] = float(val)
                except ValueError:
                    pass

except IOError:
    pass

MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
try:
    with open(MODEL_PATH, "rb") as inp:
        MODEL = pickle.load(inp)
except IOError:
    pass
