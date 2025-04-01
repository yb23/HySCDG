import numpy as np
import pandas as pd
import geopandas as gpd
import glob
from PIL import Image, ImageDraw

from .utils import convert_to_color

import rasterio
import rasterio.mask


DATA_ROOT = "/home/YBenidir/Documents/DATASETS/FLAIR1/"


REPLACE_CLASSES = np.array([2, 3, 4, 6, 7, 8, 9, 10, 11, 12])
FREQ_CLASSES = np.array([8.25, 13.72, 3.47, 2.74, 15.38, 6.95, 3.13, 17.84, 10.98, 3.88])
FREQ_CLASSES = FREQ_CLASSES / FREQ_CLASSES.sum()

REPLACE_CLASSES = np.arange(0, 20)
FREQ_CLASSES = np.array([0., 8.14, 8.25, 13.72, 3.47, 4.88, 2.74, 15.38, 6.95, 3.13, 17.84, 10.98, 3.88, 0., 0., 0., 0., 0., 0., 0.])
FREQ_CLASSES = FREQ_CLASSES / FREQ_CLASSES.sum()

CLASSES_NAMES = np.array(["","building","pervious surface","road","bare soil","water","coniferous trees","deciduous trees","brushwood","vineyard","grass","agricultural vegetation","plowed land","swimming pool","snow","cut","mixed","lignous","greenhouse",""])
PROMPTS_START = ["aerial view of ", "top view of ", "satellite view of "]


def little_mask(H=512, W=512, min_size=50, max_size=300, ellipsis=False):
    h = np.random.randint(0,H-min_size)
    w = np.random.randint(0,W-min_size)
    h1 = np.random.randint(h+min_size, min(H, h+max_size))
    w1 = np.random.randint(w+min_size, min(W, w+max_size))
    if ellipsis:
        mask = Image.new("L", (H,W), 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse((h, w, h1,w1),fill=255)
        mask = np.array(mask)[None,:]
    else:
        mask = np.zeros([1,H,W])
        mask[0,h:h1,w:w1] = 1
    return mask.astype(bool)


def getMaskedObjects(labels,mask):
    freq = np.bincount((labels[0] * mask.astype(int)).flatten(), minlength=20) / mask.sum()
    freq[0] = 0
    classes = CLASSES_NAMES[(np.nan_to_num(freq/FREQ_CLASSES)>1)]
    return " and ".join(classes)
    

wkt_2154 = """
PROJCS["RGF93 / Lambert-93",
    GEOGCS["RGF93",
        DATUM["Reseau_Geodesique_Francais_1993",
            SPHEROID["GRS 1980",6378137,298.257222101,
                AUTHORITY["EPSG","7019"]],
            AUTHORITY["EPSG","6171"]],
        PRIMEM["Greenwich",0,
            AUTHORITY["EPSG","8901"]],
        UNIT["degree",0.0174532925199433,
            AUTHORITY["EPSG","9122"]],
        AUTHORITY["EPSG","4171"]],
    PROJECTION["Lambert_Conformal_Conic_2SP"],
    PARAMETER["standard_parallel_1",49],
    PARAMETER["standard_parallel_2",44],
    PARAMETER["latitude_of_origin",46.5],
    PARAMETER["central_meridian",3],
    PARAMETER["false_easting",700000],
    PARAMETER["false_northing",6600000],
    UNIT["metre",1,
        AUTHORITY["EPSG","9001"]],
    AUTHORITY["EPSG","2154"]]
"""

CRS = rasterio.crs.CRS.from_wkt(wkt_2154)

def saveRaster(filename, img_np, transform, crs="epsg:2154",dtype=None):
    crs = CRS   ##################################################
    if dtype is None:
        dtype = img_np.dtype
    with rasterio.open(filename,'w',driver='GTiff',height=img_np.shape[1],width=img_np.shape[2],count=img_np.shape[0],dtype=dtype,crs=crs,transform=transform,compress="lzw") as dst:
        dst.write(img_np)