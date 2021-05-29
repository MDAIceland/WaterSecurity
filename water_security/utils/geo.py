import os
import urllib.request
import xml.etree.ElementTree as ET
from math import sqrt
from typing import TypedDict

import haversine as hs
import requests
from tqdm import tqdm
from tqdm.notebook import tqdm

TWO_TO_THREE_LETTER_CODE = {
    "AF": "AFG",
    "AX": "ALA",
    "AL": "ALB",
    "DZ": "DZA",
    "AS": "ASM",
    "AD": "AND",
    "AO": "AGO",
    "AI": "AIA",
    "AQ": "ATA",
    "AG": "ATG",
    "AR": "ARG",
    "AM": "ARM",
    "AW": "ABW",
    "AU": "AUS",
    "AT": "AUT",
    "AZ": "AZE",
    "BS": "BHS",
    "BH": "BHR",
    "BD": "BGD",
    "BB": "BRB",
    "BY": "BLR",
    "BE": "BEL",
    "BZ": "BLZ",
    "BJ": "BEN",
    "BM": "BMU",
    "BT": "BTN",
    "BO": "BOL",
    "BA": "BIH",
    "BW": "BWA",
    "BV": "BVT",
    "BR": "BRA",
    "IO": "IOT",
    "BN": "BRN",
    "BG": "BGR",
    "BF": "BFA",
    "BI": "BDI",
    "KH": "KHM",
    "CM": "CMR",
    "CA": "CAN",
    "CV": "CPV",
    "KY": "CYM",
    "CF": "CAF",
    "TD": "TCD",
    "CL": "CHL",
    "CN": "CHN",
    "CX": "CXR",
    "CC": "CCK",
    "CO": "COL",
    "KM": "COM",
    "CG": "COG",
    "CD": "COD",
    "CK": "COK",
    "CR": "CRI",
    "CI": "CIV",
    "HR": "HRV",
    "CU": "CUB",
    "CY": "CYP",
    "CZ": "CZE",
    "DK": "DNK",
    "DJ": "DJI",
    "DM": "DMA",
    "DO": "DOM",
    "EC": "ECU",
    "EG": "EGY",
    "SV": "SLV",
    "GQ": "GNQ",
    "ER": "ERI",
    "EE": "EST",
    "ET": "ETH",
    "FK": "FLK",
    "FO": "FRO",
    "FJ": "FJI",
    "FI": "FIN",
    "FR": "FRA",
    "GF": "GUF",
    "PF": "PYF",
    "TF": "ATF",
    "GA": "GAB",
    "GM": "GMB",
    "GE": "GEO",
    "DE": "DEU",
    "GH": "GHA",
    "GI": "GIB",
    "GR": "GRC",
    "GL": "GRL",
    "GD": "GRD",
    "GP": "GLP",
    "GU": "GUM",
    "GT": "GTM",
    "GG": "GGY",
    "GN": "GIN",
    "GW": "GNB",
    "GY": "GUY",
    "HT": "HTI",
    "HM": "HMD",
    "VA": "VAT",
    "HN": "HND",
    "HK": "HKG",
    "HU": "HUN",
    "IS": "ISL",
    "IN": "IND",
    "ID": "IDN",
    "IR": "IRN",
    "IQ": "IRQ",
    "IE": "IRL",
    "IM": "IMN",
    "IL": "ISR",
    "IT": "ITA",
    "JM": "JAM",
    "JP": "JPN",
    "JE": "JEY",
    "JO": "JOR",
    "KZ": "KAZ",
    "KE": "KEN",
    "KI": "KIR",
    "KP": "PRK",
    "KR": "KOR",
    "KW": "KWT",
    "KG": "KGZ",
    "LA": "LAO",
    "LV": "LVA",
    "LB": "LBN",
    "LS": "LSO",
    "LR": "LBR",
    "LY": "LBY",
    "LI": "LIE",
    "LT": "LTU",
    "LU": "LUX",
    "MO": "MAC",
    "MK": "MKD",
    "MG": "MDG",
    "MW": "MWI",
    "MY": "MYS",
    "MV": "MDV",
    "ML": "MLI",
    "MT": "MLT",
    "MH": "MHL",
    "MQ": "MTQ",
    "MR": "MRT",
    "MU": "MUS",
    "YT": "MYT",
    "MX": "MEX",
    "FM": "FSM",
    "MD": "MDA",
    "MC": "MCO",
    "MN": "MNG",
    "ME": "MNE",
    "MS": "MSR",
    "MA": "MAR",
    "MZ": "MOZ",
    "MM": "MMR",
    "NA": "NAM",
    "NR": "NRU",
    "NP": "NPL",
    "NL": "NLD",
    "AN": "ANT",
    "NC": "NCL",
    "NZ": "NZL",
    "NI": "NIC",
    "NE": "NER",
    "NG": "NGA",
    "NU": "NIU",
    "NF": "NFK",
    "MP": "MNP",
    "NO": "NOR",
    "OM": "OMN",
    "PK": "PAK",
    "PW": "PLW",
    "PS": "PSE",
    "PA": "PAN",
    "PG": "PNG",
    "PY": "PRY",
    "PE": "PER",
    "PH": "PHL",
    "PN": "PCN",
    "PL": "POL",
    "PT": "PRT",
    "PR": "PRI",
    "QA": "QAT",
    "RE": "REU",
    "RO": "ROU",
    "RU": "RUS",
    "RW": "RWA",
    "BL": "BLM",
    "SH": "SHN",
    "KN": "KNA",
    "LC": "LCA",
    "MF": "MAF",
    "PM": "SPM",
    "VC": "VCT",
    "WS": "WSM",
    "SM": "SMR",
    "ST": "STP",
    "SA": "SAU",
    "SN": "SEN",
    "RS": "SRB",
    "SC": "SYC",
    "SL": "SLE",
    "SG": "SGP",
    "SK": "SVK",
    "SI": "SVN",
    "SB": "SLB",
    "SO": "SOM",
    "ZA": "ZAF",
    "GS": "SGS",
    "ES": "ESP",
    "LK": "LKA",
    "SD": "SDN",
    "SR": "SUR",
    "SJ": "SJM",
    "SZ": "SWZ",
    "SE": "SWE",
    "CH": "CHE",
    "SY": "SYR",
    "TW": "TWN",
    "TJ": "TJK",
    "TZ": "TZA",
    "TH": "THA",
    "TL": "TLS",
    "TG": "TGO",
    "TK": "TKL",
    "TO": "TON",
    "TT": "TTO",
    "TN": "TUN",
    "TR": "TUR",
    "TM": "TKM",
    "TC": "TCA",
    "TV": "TUV",
    "UG": "UGA",
    "UA": "UKR",
    "AE": "ARE",
    "GB": "GBR",
    "US": "USA",
    "UM": "UMI",
    "UY": "URY",
    "UZ": "UZB",
    "VU": "VUT",
    "VE": "VEN",
    "VN": "VNM",
    "VG": "VGB",
    "VI": "VIR",
    "WF": "WLF",
    "EH": "ESH",
    "YE": "YEM",
    "ZM": "ZMB",
    "ZW": "ZWE",
}


class PlaceInfo(TypedDict):
    city: str
    country: str
    code: str


def get_elevation(latitude: float, longitude: float):
    """
    Returns coarse modeled elevation from gtopo30 for a specific latitude and longitude
    """
    req = requests.get(
        f"http://api.geonames.org/gtopo30?lat={latitude}&lng={longitude}&username=vaslem"
    )
    return int(req.text)


def get_place(latitude: float, longitude: float) -> PlaceInfo:
    """
    Returns city, country and country 3-letter code, given latitude and longitude
    Raises if the supplied coordinates cannot be matched to a specific country (eg if those refer to sea area)
    """
    req = requests.get(
        f"http://api.geonames.org/findNearbyPlaceName?lat={latitude}&lng={longitude}&cities=cities15000&username=vaslem"
    )
    tree = ET.fromstring(req.text)
    geoname = tree.find("geoname")
    try:
        city = geoname.find("toponymName").text
        country = geoname.find("countryName").text
        code = geoname.find("countryCode").text
        return {
            "city": city,
            "country": country,
            "code": TWO_TO_THREE_LETTER_CODE[code],
        }
    except AttributeError:
        raise


def is_close(loc1, loc2, thres: float = 3) -> bool:
    """
    Accepts 2 points defined by 2 coordinates (iterables of size 2) and based on a distance threshold (in km),
    returns whether those points are close or not to each other
    """
    return hs.haversine(loc1, loc2) < thres


import rasterio
import numpy as np

## Answer to
## https://stackoverflow.com/questions/60127026/python-how-do-i-get-the-pixel-values-from-an-geotiff-map-by-coordinate
def get_coordinate_pixel(
    map: str, lat: float, lon: float, pixels_width: int, pixels_height: int, crs="WGS84"
) -> np.ndarray:
    """
    Given a geotif map, open it,
    read the pixels values that correspond to the provided latitude and longitude,
    with a specific bounding box pixels_width, pixels_height, and return them
    """
    # open map
    with rasterio.open(map, crs=crs) as dataset:
        # get pixel x+y of the coordinate
        py, px = dataset.index(lat, lon)
        # create 1x1px window of the pixel
        window = rasterio.windows.Window(
            px - pixels_width // 2, py - pixels_height // 2, pixels_width, pixels_height
        )
        # read rgb values of the window
        return dataset.read(window=window)


CHECKED_DENSITY_SOURCE = False


def check_1k_population_density_source():
    global CHECKED_DENSITY_SOURCE
    from data.unlabeled import POPULATION_DENSITY_PATH, POPULATION_DENSITY_URL

    if not CHECKED_DENSITY_SOURCE:
        try:
            get_coordinate_pixel(
                POPULATION_DENSITY_PATH, 0, 0, pixels_width=3, pixels_height=3
            )

            rasterio.open(POPULATION_DENSITY_PATH)
            CHECKED_DENSITY_SOURCE = True
        except:
            pass
    if not CHECKED_DENSITY_SOURCE:
        print("Population Density Map is not available, it will be downloaded")
        from utils.download import download_url

        try:
            return download_url(POPULATION_DENSITY_URL, POPULATION_DENSITY_PATH)
        except KeyboardInterrupt:
            try:
                os.remove(POPULATION_DENSITY_PATH)
            except:
                pass


def get_average_1k_population_density(longitude: float, latitude: float) -> int:
    """
    Based on provided longitude and latitude, get the median population density, as computed
    in a 7x7 square around the pixel of the population density geotif map located in  POPULATION_DENSITY_PATH.
    """
    check_1k_population_density_source()
    from data.unlabeled import POPULATION_DENSITY_PATH

    oret = get_coordinate_pixel(
        POPULATION_DENSITY_PATH, latitude, longitude, pixels_width=7, pixels_height=7
    ).squeeze()
    cnt = 3
    ret = oret
    while cnt:
        res = np.median(ret[ret > 0])
        if res > 1:
            return res
        cnt -= 1
        ret = ret[1:-1, -1:1]
    return np.median(oret[oret > 0])
