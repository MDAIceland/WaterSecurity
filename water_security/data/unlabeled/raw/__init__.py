import json
import os

import pandas as pd

_cdir = os.sep.join(os.path.split(__file__)[:-1])
try:
    hdro_path = os.path.join(_cdir, "hdro2019.json")
    with open(hdro_path) as jsonfile:
        json_hdro = json.load(jsonfile)
    hdro_inicator_values = json_hdro["indicator_value"]
    hdro_country_name = json_hdro["country_name"]
    hdro_indicator_name = json_hdro["indicator_name"]
except Exception as e:
    print("Something went wrong loading the Human Development Dataset (hdro)", e)

try:
    econ_co_path = os.path.join(_cdir, "Economic_Fitness_CSV", "Country.csv")
    econ_da_path = os.path.join(_cdir, "Economic_Fitness_CSV", "Data.csv")
    econ_se_path = os.path.join(_cdir, "Economic_Fitness_CSV", "Series.csv")
    econ_co = pd.read_csv(econ_co_path)
    econ_da = pd.read_csv(econ_da_path)
    econ_se = pd.read_csv(econ_se_path)
except Exception as e:
    print("Something went wrong loading the Economic Fitness Dataset", e)


try:
    edstats_dir = os.path.join(_cdir, "Edstats_csv")
    if not os.path.isdir(edstats_dir) or (len(os.listdir(edstats_dir)) != 5):
        from zipfile import ZipFile

        # Create a ZipFile Object and load sample.zip in it
        with ZipFile(f"{edstats_dir}.zip", "r") as zipObj:
            # Extract all the contents of zip file in current directory
            os.makedirs(edstats_dir, exist_ok=True)
            zipObj.extractall(edstats_dir)
    edstats_co_path = os.path.join(edstats_dir, "EdStatsCountry.csv")
    edstats_da_path = os.path.join(edstats_dir, "EdStatsData.csv")
    edstats_se_path = os.path.join(edstats_dir, "EdStatsSeries.csv")
    edstats_co = pd.read_csv(edstats_co_path)
    edstats_da = pd.read_csv(edstats_da_path)
    edstats_se = pd.read_csv(edstats_se_path)
except Exception as e:
    print("Something went wrong loading the Education Dataset", e)

try:
    aquastat_eah_path = os.path.join(_cdir, "aquastat", "aquastat_env_and_health.csv")
    aquastat_wr_path = os.path.join(_cdir, "aquastat", "aquastat_water_resources.csv")
    aquastat_wu_path = os.path.join(_cdir, "aquastat", "aquastat_water_use.csv")
    aquastat_cc_path = os.path.join(_cdir, "aquastat", "aquastat_country_code.csv")
    aquastat_eah = pd.read_csv(aquastat_eah_path, skipfooter=8)
    aquastat_wr = pd.read_csv(aquastat_wr_path, skipfooter=8)
    aquastat_wu = pd.read_csv(aquastat_wu_path, skipfooter=8)
    aquastat_eah["Md"] = aquastat_eah["Unknown"]
    aquastat_wr["Md"] = aquastat_wr["Unknown"]
    aquastat_wu["Md"] = aquastat_wu["Unknown"]
    aquastat_eah.drop("Unknown", axis=1, inplace=True)
    aquastat_wr.drop("Unknown", axis=1, inplace=True)
    aquastat_wu.drop("Unknown", axis=1, inplace=True)
    aquastat_cc = pd.read_csv(aquastat_cc_path)
except Exception as e:
    print("Something went wrong loading the Aquastat Dataset", e)
