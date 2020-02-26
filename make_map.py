from sys import argv
from pathlib import Path
from loguru import logger
from typing import Tuple, Dict, Union, Optional
import json

import numpy as np
import pandas as pd
import geopandas as gpd
from matplotlib import pyplot as plt

from pyproj import CRS, Transformer
from PIL import Image, ImageFont
from PIL.ImageDraw import Draw
from io import BytesIO
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import transform

MercatorBbox = Tuple[float]

def parse_config(config_path: Path) -> Dict[str, Union[int, float, str]]:
    config_path = Path(config_path)
    logger.info(f"Reading config from {config_path}")

    with config_path.open("r") as f:
        config = json.load(f)

    if "subtitle" not in config:
        config["subtitle"] = None

    logger.info("Done!")
    return config

def make_font(path: Path, size: float, y_res: int):
    font_path = Path(path)
    if not font_path.exists():
        raise FileNotFoundError(f"Did not find {path}!")
    font_size = int(size/6144*y_res)
    str_font_path = str(font_path.absolute())
    return ImageFont.truetype(str_font_path, size=font_size)


def load_data(data_path: Path) -> gpd.GeoDataFrame:
    data_path = Path(data_path)
    logger.info(f"Reading data from {data_path}...")
    df = gpd.read_file(str(data_path.absolute()))
    logger.info(f"Projecting data to EPSG:3395...")
    p_df = df.to_crs(epsg=3395)
    logger.info(f"Building spatial index...")
    p_df.sindex
    logger.info(f"Done!")
    return p_df

def make_mercator_bbox(lng: float, lat: float, h_radius: float, v_radius: float) ->MercatorBbox:
    tx = Transformer.from_crs('EPSG:4326', 'EPSG:3395', always_xy=True)
    cx, cy = tx.transform(lng, lat)
    p_w, p_e = cx - h_radius, cx + h_radius
    p_s, p_n = cy - v_radius, cy + v_radius
    p_bbox = p_w, p_s, p_e, p_n
    assert p_w <= p_e
    assert p_s <= p_n
    return p_bbox

def get_geometry(bbox: MercatorBbox, df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    possible_matches = list(df.sindex.intersection(bbox))
    p_sub_df = df.iloc[possible_matches]
    return p_sub_df.geometry

def rasterize_geometry(geometry: gpd.GeoSeries, bbox: MercatorBbox, y_res: int):
    w, s, e, n = bbox
    assert w <= e
    assert s <= n
    sf = (e-w)/(n-s)
    x_res = int(sf*y_res)

    def imgspace_transform(xs, ys):
        """
        Given a set of x and y coordinates, convert them to pixels within the current image.
        """
        nonlocal w, s, e, n, x_res, y_res
        xs = np.array(xs)
        ys = np.array(ys)
        xs -= w
        ys -= s
        xs /= e - w
        ys /= n - s
        xs *= x_res
        ys *= y_res
        ys = y_res-ys
        return xs.astype(np.int64), ys.astype(np.int64)

    img_geometry = [transform(imgspace_transform, shape) for shape in geometry]
    img_geometry = [list(poly) if type(poly) == MultiPolygon else poly for poly in img_geometry]
    tmp_img_geometry = []
    for poly in img_geometry:
        if type(poly) == list:
            tmp_img_geometry += poly
        else:
            tmp_img_geometry += [poly]
    img_geometry = tmp_img_geometry

    img = Image.new("L", (x_res, y_res))
    draw = Draw(img)
    for polygon in img_geometry:
        if type(polygon) != Polygon:
            logger.warning(f"Skipping non-polygon {type(polygon)}!")
            continue
        draw.polygon(list(polygon.exterior.coords), fill=1)
        for interior_hole in polygon.interiors:
            draw.polygon(list(interior_hole.coords), fill=0)

    return img, draw

def dms(cx: float, cy: float):
    ordx = "E" if cx >= 0 else "W"
    ordy = "N" if cy >= 0 else "S"    
    cx, cy = abs(cx), abs(cy)
    degx, degy = int(cx), int(cy)
    cx, cy = cx - degx, cy - degy
    minx, miny = int(60*cx), int(60*cy)
    cx, cy = cx - minx/60, cy - miny/60
    secx, secy = int(3600*cx), int(3600*cy)
    return f"{degy}° {miny}' {secy}\" {ordy}, {degx}° {minx}' {secx}\" {ordx}"

#TODO: finish me
def add_text(img: Image, draw: Draw, city_name: str, dms_str:str, title_font, subtitle_font, subtitle: Optional[str] = None):
    im_h, im_w = np.array(img).shape # pylint: disable=unpacking-non-sequence
    box_start_offset = (650/6144)*im_h
    box_end_offset = (425/6144)*im_h
    title_offset = (650/6144)*im_h
    subtitle1_offset = (500/6144)*im_h
    subtitle2_offset = (450/6144)*im_h
    box_x_start = (150/6144)*im_h
    box_width = 100*len(" ".join(city_name))/6144*im_h

    name = " ".join(city_name.upper())
    draw.rectangle(((0, im_h-box_start_offset), (box_width, im_h-box_end_offset)), fill=0, outline=1)
    draw.text((box_x_start, im_h-title_offset), name, fill=1, font=title_font)
    draw.text((box_x_start, im_h-subtitle1_offset), dms_str, fill=1, font=subtitle_font)
    if subtitle is not None:
        draw.text((box_x_start, im_h-subtitle2_offset), subtitle, fill=1, font=subtitle_font)

def convert_for_lasing(img: Image) -> np.ndarray:
    img = np.array(img)
    h, w = img.shape
    lase_mask = np.zeros((h, w, 3), np.uint8)
    lase_mask[:, :, 0] = img*255
    lase_mask[:3, :, 1] = 255
    lase_mask[-3:, :, 1] = 255
    lase_mask[:, :3, 1] = 255
    lase_mask[:, -3:, 1] = 255
    return Image.fromarray(lase_mask)

if __name__ == "__main__":
    _, config_path = argv
    config = parse_config(config_path)
    df = load_data(config['data_file'])
    lng, lat = config['center']
    logger.info(f"Selecting AOI centered at x={lng}, y={lat}")
    bbox = make_mercator_bbox(lng, lat, config['horiz_radius'], config['vert_radius'])
    geom = get_geometry(bbox, df)
    logger.info(f"Rendering {len(geom)} polygons...")
    img, draw = rasterize_geometry(geom, bbox, config['y_res'])
    title_font = make_font(config['title_font'], config['title_size'], config['y_res'])
    subtitle_font = make_font(config['subtitle_font'], config['subtitle_size'], config['y_res'])
    dms_string = dms(lng, lat)
    add_text(img, draw, config['city_name'], dms_string, title_font, subtitle_font, config['subtitle'])
    img = convert_for_lasing(img)
    logger.info(f"Writing to {config['output_name']}")
    output_folder = Path("maps/")
    output_file = output_folder / f"{config['output_name']}.png"
    output_folder.mkdir(exist_ok=True)
    img.save(str(output_file.absolute()))