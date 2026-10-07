"""
===============================================================================
File: src/geospatial/geo_utils.py
Purpose: Geospatial Metadata Extraction, Safe Raster I/O, and Geodesic Area Calculation
===============================================================================
"""

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window

from src.utils.config import CLASS_COLORMAP

# Default data paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
RAW_S1_PATH = BASE_DIR / "data" / "raw" / "sentinel1" / "Sentinel1_VV_VH.tif"
RAW_S2_PATH = BASE_DIR / "data" / "raw" / "sentinel2" / "Sentinel2_Bands.tif"
PROCESSED_S1_PATH = BASE_DIR / "data" / "processed" / "sentinel1" / "sentinel1_processed.tif"
PROCESSED_S2_PATH = BASE_DIR / "data" / "processed" / "sentinel2" / "sentinel2_processed.tif"
PROCESSED_LABELS_PATH = BASE_DIR / "data" / "processed" / "labels" / "worldcover_labels.tif"


def hex_to_rgb(hex_str: str) -> Tuple[int, int, int]:
    """Converts hex color string to (R, G, B) tuple of ints in [0, 255]."""
    hex_str = hex_str.lstrip('#')
    return tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))


def colorize_mask(mask: np.ndarray, colormap: Optional[Dict[int, str]] = None) -> np.ndarray:
    """
    Renders a 2D categorical mask array (H, W) into an RGB image array (H, W, 3) in uint8 [0, 255].
    Preserves sharp integer boundaries without interpolation blurring.
    """
    if colormap is None:
        colormap = CLASS_COLORMAP

    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for class_idx, hex_color in colormap.items():
        r, g, b = hex_to_rgb(hex_color)
        match = (mask == class_idx)
        rgb[match] = [r, g, b]
    return rgb



def get_raster_metadata(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Extracts complete, verified geospatial metadata from any GeoTIFF raster.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        return {"exists": False, "filename": filepath.name, "error": "File not found"}

    try:
        with rasterio.open(filepath) as src:
            bounds = src.bounds
            center_lat = (bounds.bottom + bounds.top) / 2.0
            center_lon = (bounds.left + bounds.right) / 2.0
            crs_str = str(src.crs) if src.crs else "Unknown"

            res_x, res_y = src.res

            descriptions = list(src.descriptions) if src.descriptions else []

            return {
                "exists": True,
                "filename": filepath.name,
                "filepath": str(filepath),
                "width": src.width,
                "height": src.height,
                "count": src.count,
                "dtypes": [str(d) for d in src.dtypes],
                "crs": crs_str,
                "transform": [float(x) for x in src.transform][:6],
                "resolution_deg": (float(res_x), float(res_y)),
                "bounds": {
                    "left": float(bounds.left),
                    "bottom": float(bounds.bottom),
                    "right": float(bounds.right),
                    "top": float(bounds.top),
                },
                "center": {"lat": float(center_lat), "lon": float(center_lon)},
                "nodata": src.nodata,
                "band_descriptions": descriptions,
            }
    except Exception as e:
        return {"exists": True, "filename": filepath.name, "error": str(e)}


def calculate_pixel_area_m2(center_lat_deg: float, res_x_deg: float, res_y_deg: float) -> float:
    """
    Calculates exact geodesic area of a single pixel on WGS84 ellipsoid (EPSG:4326) in m^2.
    """
    # WGS-84 ellipsoid constants
    a = 6378137.0  # semi-major axis in meters
    e2 = 0.00669437999014  # first eccentricity squared

    phi = math.radians(center_lat_deg)
    sin_phi = math.sin(phi)

    # Meridional radius of curvature (North-South)
    m = a * (1 - e2) / ((1 - e2 * sin_phi**2) ** 1.5)
    # Prime vertical radius of curvature (East-West)
    n = a / math.sqrt(1 - e2 * sin_phi**2)

    deg_to_rad = math.pi / 180.0
    dx_meters = n * math.cos(phi) * (res_x_deg * deg_to_rad)
    dy_meters = m * (res_y_deg * deg_to_rad)

    return abs(dx_meters * dy_meters)


def calculate_class_areas(
    pixel_counts: Dict[Union[int, str], int],
    center_lat_deg: float = 14.464,
    res_x_deg: float = 8.983152841195215e-05,
    res_y_deg: float = 8.983152841195215e-05,
) -> Dict[str, Dict[str, float]]:
    """
    Computes geodesic land-cover area (hectares and km^2) from pixel counts.
    """
    pixel_area_m2 = calculate_pixel_area_m2(center_lat_deg, res_x_deg, res_y_deg)

    results = {}
    for cls_name, count in pixel_counts.items():
        area_m2 = float(count) * pixel_area_m2
        results[str(cls_name)] = {
            "pixels": count,
            "area_m2": area_m2,
            "hectares": round(area_m2 / 10000.0, 2),
            "km2": round(area_m2 / 1000000.0, 4),
        }
    return results


def percentile_stretch(
    img: np.ndarray,
    p_low: float = 2.0,
    p_high: float = 98.0,
) -> np.ndarray:
    """
    Robust 2-98% percentile contrast stretch for satellite visualization.
    Preserves original data representation, modifying only the visual array.
    """
    valid = np.isfinite(img)
    if not np.any(valid):
        return np.zeros_like(img, dtype=np.float32)

    if img.ndim == 3 and img.shape[2] in (3, 4):  # Multi-channel HWC
        stretched = np.zeros_like(img, dtype=np.float32)
        for c in range(img.shape[2]):
            channel = img[:, :, c]
            v_chan = channel[np.isfinite(channel)]
            if v_chan.size > 0:
                p2 = np.percentile(v_chan, p_low)
                p98 = np.percentile(v_chan, p_high)
                diff = p98 - p2
                if diff <= 0:
                    diff = 1.0
                stretched[:, :, c] = np.clip((channel - p2) / diff, 0.0, 1.0)
        return stretched
    else:  # Single channel 2D
        v_pix = img[valid]
        p2 = np.percentile(v_pix, p_low)
        p98 = np.percentile(v_pix, p_high)
        diff = p98 - p2
        if diff <= 0:
            diff = 1.0
        return np.clip((img - p2) / diff, 0.0, 1.0).astype(np.float32)


def load_sentinel2_native_overview(
    s2_path: Optional[Path] = None,
    mode: str = "RGB",
    max_dim: int = 1024,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Reads a memory-safe overview of the native Sentinel-2 GeoTIFF.
    Modes:
      - 'RGB': B4 (Red), B3 (Green), B2 (Blue) -> indices (3, 2, 1)
      - 'False Color': B8 (NIR), B4 (Red), B3 (Green) -> indices (4, 3, 2)
      - 'SWIR': B12 (SWIR-2), B8 (NIR), B4 (Red) -> indices (6, 4, 3)
      - 'B2', 'B3', 'B4', 'B8', 'B11', 'B12': Single band
    """
    if s2_path is None or not Path(s2_path).exists():
        s2_path = PROCESSED_S2_PATH if PROCESSED_S2_PATH.exists() else RAW_S2_PATH

    s2_path = Path(s2_path)
    meta = get_raster_metadata(s2_path)

    # Band mapping in 6-band processed/raw raster:
    # 1: B2 (Blue), 2: B3 (Green), 3: B4 (Red), 4: B8 (NIR), 5: B11 (SWIR-1), 6: B12 (SWIR-2)
    band_map = {
        "B2": 1,
        "B3": 2,
        "B4": 3,
        "B8": 4,
        "B11": 5,
        "B12": 6,
    }

    with rasterio.open(s2_path) as src:
        # Calculate overview downsample factor for responsive browser rendering
        scale = max(1, math.ceil(max(src.width, src.height) / max_dim))
        out_shape = (src.height // scale, src.width // scale)

        if mode == "RGB":
            b4 = src.read(3, out_shape=out_shape, resampling=Resampling.bilinear)
            b3 = src.read(2, out_shape=out_shape, resampling=Resampling.bilinear)
            b2 = src.read(1, out_shape=out_shape, resampling=Resampling.bilinear)
            raw_composite = np.dstack((b4, b3, b2))
            vis = percentile_stretch(raw_composite, 2.0, 98.0)
        elif mode == "False Color":
            b8 = src.read(4, out_shape=out_shape, resampling=Resampling.bilinear)
            b4 = src.read(3, out_shape=out_shape, resampling=Resampling.bilinear)
            b3 = src.read(2, out_shape=out_shape, resampling=Resampling.bilinear)
            raw_composite = np.dstack((b8, b4, b3))
            vis = percentile_stretch(raw_composite, 2.0, 98.0)
        elif mode == "SWIR":
            b12 = src.read(6, out_shape=out_shape, resampling=Resampling.bilinear)
            b8 = src.read(4, out_shape=out_shape, resampling=Resampling.bilinear)
            b4 = src.read(3, out_shape=out_shape, resampling=Resampling.bilinear)
            raw_composite = np.dstack((b12, b8, b4))
            vis = percentile_stretch(raw_composite, 2.0, 98.0)
        elif mode in band_map:
            band_idx = band_map[mode]
            raw_b = src.read(band_idx, out_shape=out_shape, resampling=Resampling.bilinear)
            vis = percentile_stretch(raw_b, 2.0, 98.0)
        else:
            raise ValueError(f"Unsupported Sentinel-2 mode: {mode}")

    return vis, meta


def load_sentinel1_native_overview(
    s1_processed_path: Optional[Path] = None,
    s1_raw_path: Optional[Path] = None,
    band: str = "VV",
    max_dim: int = 1024,
) -> Tuple[np.ndarray, Dict[str, Any], str]:
    """
    Safely reads Sentinel-1 SAR imagery with graceful fallback from corrupted processed TIFF
    to raw reference TIFF, preventing unhandled exceptions.
    Returns (visualization_array, metadata, data_source_description).
    """
    if s1_processed_path is None:
        s1_processed_path = PROCESSED_S1_PATH
    if s1_raw_path is None:
        s1_raw_path = RAW_S1_PATH

    s1_target = None
    source_desc = ""

    # Attempt processed raster first
    if Path(s1_processed_path).exists():
        try:
            with rasterio.open(s1_processed_path) as test_src:
                _ = test_src.read(1, window=Window(0, 0, 10, 10))
            s1_target = Path(s1_processed_path)
            source_desc = "Processed Sentinel-1 GeoTIFF (Normalized [0, 1])"
        except Exception:
            s1_target = None

    # Fallback to uncorrupted raw reference raster
    if s1_target is None and Path(s1_raw_path).exists():
        s1_target = Path(s1_raw_path)
        source_desc = "Raw Sentinel-1 GeoTIFF Fallback (dB Backscatter with Live Percentile Stretch)"

    if s1_target is None:
        raise FileNotFoundError("Neither processed nor raw Sentinel-1 GeoTIFF could be opened.")

    meta = get_raster_metadata(s1_target)

    with rasterio.open(s1_target) as src:
        scale = max(1, math.ceil(max(src.width, src.height) / max_dim))
        out_shape = (src.height // scale, src.width // scale)

        if band == "VV":
            raw_arr = src.read(1, out_shape=out_shape, resampling=Resampling.bilinear)
            vis = percentile_stretch(raw_arr, 1.0, 99.0)
        elif band == "VH":
            raw_arr = src.read(2, out_shape=out_shape, resampling=Resampling.bilinear)
            vis = percentile_stretch(raw_arr, 1.0, 99.0)
        elif band == "VV/VH Ratio":
            vv = src.read(1, out_shape=out_shape, resampling=Resampling.bilinear)
            vh = src.read(2, out_shape=out_shape, resampling=Resampling.bilinear)
            # If in dB: difference VV - VH corresponds to ratio in linear scale
            # If normalized: ratio with epsilon
            if "raw" in source_desc.lower() or np.nanmin(vv) < -5:
                # In dB: VV (dB) - VH (dB) = 10*log10(VV_lin / VH_lin)
                ratio = vv - vh
            else:
                ratio = np.divide(vv, vh + 1e-4, out=np.zeros_like(vv), where=vh > 1e-4)
            vis = percentile_stretch(ratio, 2.0, 98.0)
        else:
            raise ValueError(f"Unsupported Sentinel-1 band: {band}")

    return vis, meta, source_desc


def export_classification_geotiff(
    prediction_arr: np.ndarray,
    output_path: Union[str, Path],
    bounds: Dict[str, float],
    crs: str = "EPSG:4326",
) -> Path:
    """
    Exports a 2D classification prediction map to a standard georeferenced GeoTIFF.
    Preserves exact geographic transform, CRS, and categorical integer encoding.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    height, width = prediction_arr.shape
    transform = rasterio.transform.from_bounds(
        west=bounds["left"],
        south=bounds["bottom"],
        east=bounds["right"],
        north=bounds["top"],
        width=width,
        height=height,
    )

    profile = {
        "driver": "GTiff",
        "dtype": rasterio.uint8,
        "nodata": 255,
        "width": width,
        "height": height,
        "count": 1,
        "crs": crs,
        "transform": transform,
        "compress": "lzw",
    }

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(prediction_arr.astype(np.uint8), 1)
        dst.set_band_description(1, "FusionLand_AI_Classification")

    return output_path


def inspect_pixel(filepath: Union[str, Path], row: int, col: int) -> Dict[str, Any]:
    """
    Directly queries a raster pixel using true Affine georeferencing.
    Never hard-codes coordinates: transforms (col, row) -> (longitude, latitude)
    using the embedded raster transform.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        return {"valid": False, "error": f"Raster file '{filepath.name}' not found."}

    try:
        with rasterio.open(filepath) as src:
            if not (0 <= row < src.height and 0 <= col < src.width):
                return {
                    "valid": False,
                    "error": f"Pixel index ({row}, {col}) out of bounds for raster dimensions ({src.height}, {src.width}).",
                }

            # Geographic coordinates via affine transform
            lon, lat = src.xy(row, col)

            # Read 1x1 window for all bands
            win = Window(col, row, 1, 1)
            raw_data = src.read(window=win)  # [bands, 1, 1]

            band_vals = [float(raw_data[b, 0, 0]) for b in range(src.count)]

            return {
                "valid": True,
                "row": int(row),
                "col": int(col),
                "lon": float(lon),
                "lat": float(lat),
                "crs": str(src.crs) if src.crs else "Unknown",
                "bands": band_vals,
                "count": src.count,
                "descriptions": list(src.descriptions) if src.descriptions else [],
            }
    except Exception as e:
        return {"valid": False, "error": str(e)}


def inspect_coordinate(filepath: Union[str, Path], lat: float, lon: float) -> Dict[str, Any]:
    """
    Converts geographic coordinates (latitude, longitude) into raster pixel (row, col)
    using the genuine Affine inverse transform and extracts band values.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        return {"valid": False, "error": f"Raster file '{filepath.name}' not found."}

    try:
        with rasterio.open(filepath) as src:
            b = src.bounds
            if not (b.bottom <= lat <= b.top and b.left <= lon <= b.right):
                return {
                    "valid": False,
                    "error": f"Coordinate ({lat:.4f}N, {lon:.4f}E) is outside raster bounds [{b.bottom:.4f}..{b.top:.4f}, {b.left:.4f}..{b.right:.4f}].",
                }

            row, col = src.index(lon, lat)
            return inspect_pixel(filepath, row, col)
    except Exception as e:
        return {"valid": False, "error": str(e)}

