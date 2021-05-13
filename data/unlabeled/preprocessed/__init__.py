import pandas as pd
import os

_cdir = os.sep.join(os.path.split(__file__)[:-1])
try:
    econ_path = os.path.join(_cdir, "economy_preprocessed.csv")
    aqua_path = os.path.join(_cdir, "aquastat_preprocessed.csv")
    edu_path = os.path.join(_cdir, "edstats_preprocessed.csv")
    humdev_path = os.path.join(_cdir, "hdro_preprocessed.csv")
    econ = pd.read_csv(econ_path, index_col=0)
    aqua = pd.read_csv(aqua_path, index_col=0)
    edu = pd.read_csv(edu_path, index_col=0)
    humdev = pd.read_csv(humdev_path, index_col=0)
except Exception as e:
    print("Something went wrong loading the Preprocessed Dataset", e)
