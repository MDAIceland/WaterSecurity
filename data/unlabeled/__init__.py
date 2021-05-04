import pandas as pd
import json
import os

_cdir = os.sep.join(os.path.split(__file__)[:-1])
try:
    hdro_path = os.path.join(_cdir, "hdro2019.json")
    with open(hdro_path) as jsonfile:
        json_hdro = json.load(jsonfile)
    hdro_inicator_values = json_hdro['indicator_value']
    hdro_country_name = json_hdro['country_name']
    hdro_indicator_name = json_hdro['indicator_name']
except Exception as e:
    print("Something went wrong loading the Human Development Dataset (hdro)",e)

try:
    econ_co_path = os.path.join(_cdir, "Economic_Fitness_CSV\Country.csv")
    econ_da_path = os.path.join(_cdir, "Economic_Fitness_CSV\Data.csv")
    econ_se_path = os.path.join(_cdir, "Economic_Fitness_CSV\Series.csv")
    econ_co = pd.read_csv(econ_co_path)
    econ_da = pd.read_csv(econ_da_path)
    econ_se = pd.read_csv(econ_se_path)
except Exception as e:
    print("Something went wrong loading the Economic Fitness Dataset", e)

try:
    edstats_co_path = os.path.join(_cdir, "Edstats_csv/EdStatsCountry.csv")
    edstats_da_path = os.path.join(_cdir, "Edstats_csv/EdStatsData.csv")
    edstats_se_path = os.path.join(_cdir, "Edstats_csv/EdStatsSeries.csv")
    edstats_co = pd.read_csv(edstats_co_path)
    edstats_da = pd.read_csv(edstats_da_path)
    edstats_se = pd.read_csv(edstats_se_path)
except Exception as e:
    print("Something went wrong loading the Education Dataset", e)