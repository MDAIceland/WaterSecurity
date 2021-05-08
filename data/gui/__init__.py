import os
import json
from bokeh.models import GeoJSONDataSource

GUI_DIR = os.sep.join(os.path.split(__file__)[:-1])
with open(os.path.join(GUI_DIR, "countries.geojson"), "r") as inp:
    COUNTRIES_MAP_SOURCE = GeoJSONDataSource(geojson=inp.read())
from bokeh.models import ColumnDataSource
from data.labeled import LABELED_CITIES

LABELED_CITIES_SOURCE = ColumnDataSource(data=LABELED_CITIES)
