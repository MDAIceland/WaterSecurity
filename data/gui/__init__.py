import os
from bokeh.models import GeoJSONDataSource

_cdir = os.sep.join(os.path.split(__file__)[:-1])

COUNTRIES_MAP = GeoJSONDataSource(geojson=os.path.join(_cdir, "countries.geojson"))
