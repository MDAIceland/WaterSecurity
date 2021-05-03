import pandas as pd
import os

_cdir = os.sep.join(os.path.split(__file__)[:-1])
CWA_PATH = os.path.join(_cdir, "2018_-_Cities_Water_Actions.csv")
CWR_PATH = os.path.join(_cdir, "2018_-_Cities_Water_Risks.csv")
CWA = pd.read_csv(CWA_PATH)
CWR = pd.read_csv(CWR_PATH)
