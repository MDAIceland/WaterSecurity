from math import sqrt
import requests

import xml.etree.ElementTree as ET


def get_place(latitude, longitude):
    req = requests.get(
        f"http://api.geonames.org/findNearbyPlaceName?lat={latitude}&lng={longitude}&username=vaslem"
    )
    tree = ET.fromstring(req.text)
    geoname = tree.find("geoname")
    try:
        city = geoname.find("toponymName").text
        country = geoname.find("countryName").text
        code = geoname.find("countryCode").text
        return {"city": city, "country": country, "code": code}
    except AttributeError:
        raise


def is_close(p1, p2, thres=0.3):
    if sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) < thres:
        return True
    return False
