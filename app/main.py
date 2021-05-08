import ptvsd
import sys
import os

if "BOKEH_VS_DEBUG" in os.environ and os.environ["BOKEH_VS_DEBUG"] == "true":
    # 5678 is the default attach port in the VS Code debug configurations
    print("Waiting for debugger attach")
    ptvsd.enable_attach(address=("localhost", 5678), redirect_output=True)
    ptvsd.wait_for_attach()


sys.path.append(".")
from data.gui import COUNTRIES_MAP_SOURCE, LABELED_CITIES_SOURCE
from data.labeled import RISKS_MAPPING
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.plotting import curdoc
from copy import deepcopy as copy


def onlineCallback(attr, old, new):
    new["x"], new["y"]
    print("python changed selected", new)


onlinePredictionSource = ColumnDataSource(data=copy(dict(LABELED_CITIES_SOURCE.data)))
onlinePredictionSource.data = {}

onlinePredictionSource.on_change("data", onlineCallback)

from bokeh.models import (
    ColumnDataSource,
    HoverTool,
    PointDrawTool,
)
from bokeh.plotting import figure

from bokeh.events import Tap

# Input GeoJSON source that contains features for plotting
countriesSource = COUNTRIES_MAP_SOURCE
citiesSource = LABELED_CITIES_SOURCE


p = figure(
    title="Water Risks Prediction",
    plot_height=800,
    plot_width=1200,
    toolbar_location="below",
    tools="pan, wheel_zoom, box_zoom, reset",
)
p.xgrid.grid_line_color = None
p.ygrid.grid_line_color = None
# Add patch renderer to figure.
countries = p.patches(
    "xs",
    "ys",
    source=countriesSource,
    fill_color="grey",
    line_color="black",
    line_width=0.25,
    fill_alpha=1,
)
cities = p.circle(
    "longitude", "latitude", source=citiesSource, size=5, color="black", alpha=0.7
)
onlinePredictions = p.circle(
    "x", "y", source=onlinePredictionSource, size=5, color="red", alpha=0.7
)


def callback(event):
    print(onlinePredictionSource.data)


draw_tool = PointDrawTool(
    renderers=[onlinePredictions],
)
p.toolbar.active_tap = draw_tool


# Create hover tool
p.add_tools(
    HoverTool(
        renderers=[cities, onlinePredictions],
        tooltips=[("city", "@city"), ("country", "@country")]
        + [(description, f"@{risk}") for risk, description in RISKS_MAPPING.items()],
    ),
    draw_tool,
)
p.on_event(Tap, callback)
curdoc().add_root(p)
