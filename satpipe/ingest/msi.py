
"""
INGESTION: Sentinel-2 L2A ingestion (COG or JP2) with on-the-fly resampling
----------------------------------------------------------------
• Accepts scene_id  T35NND/2024/02/15   (leading “T” optional)
• Searches Earth-Search STAC, falls back to most-recent ≤ date
• Downloads 4×10 m bands (B02, B03, B04, B08)
• If any band arrives at 20 m it is upsampled to the reference 10 m grid
• Produces consolidated Zarr ready for DVC + MLflow
"""

from __future__ import annotations

import os
import tempfile
from typing import Dict

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject
import requests
import xarray as xr
from pystac_client import Client
from tqdm import tqdm

from satpipe.ingest.base import AbstractIngestor
from satpipe.utils.io import write_zarr

STAC_API = "https://earth-search.aws.element84.com/v1"
COLLECTION = "sentinel-2-l2a"          # or 'sentinel-2-l2a-cogs'

# logical → possible STAC asset keys
BANDS: Dict[str, list[str]] = {
    "B02": ["B02", "B02_JP2", "blue",  "blue-jp2"],
    "B03": ["B03", "B03_JP2", "green", "green-jp2"],
    "B04": ["B04", "B04_JP2", "red",   "red-jp2"],
    "B08": ["B08", "B08_JP2", "nir08", "nir08-jp2", "nir", "nir-jp2"],
}


def _download(url: str, dst: str, desc: str) -> None:
    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    with open(dst, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, unit_divisor=1024,
        desc=desc, leave=False
    ) as bar:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))


def _resample(src_path: str, ref_profile: rasterio.profiles.Profile) -> np.ndarray:
    """Reproject src band into the reference grid defined by ref_profile."""
    with rasterio.open(src_path) as src:
        src_arr = src.read(1)
        dst_arr = np.empty((ref_profile["height"], ref_profile["width"]),
                           dtype=src_arr.dtype)
        reproject(
            source=src_arr,
            destination=dst_arr,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref_profile["transform"],
            dst_crs=ref_profile["crs"],
            resampling=Resampling.nearest,
        )
    return dst_arr


class MSIIngestor(AbstractIngestor):
    # ------------------------------------------------------------------ #
    def download(self, scene_id: str, **_) -> Dict[str, dict]:
        tile_raw, y, m, d = scene_id.split("/")
        tile_id = tile_raw.lstrip("Tt")
        date    = f"{y}-{m}-{d}"

        stac = Client.open(STAC_API)

        # exact-date search then fallback
        items = list(
            stac.search(
                collections=[COLLECTION],
                query={"s2:mgrs_tile": {"eq": tile_id}},
                datetime=f"{date}T00:00:00Z/{date}T23:59:59Z",
                max_items=1,
            ).items()
        ) or list(
            stac.search(
                collections=[COLLECTION],
                query={"s2:mgrs_tile": {"eq": tile_id}},
                datetime=f"../{date}T23:59:59Z",
                sortby=[{"field": "properties.datetime", "direction": "desc"}],
                max_items=1,
            ).items()
        )
        if not items:
            raise ValueError(f"No Sentinel-2 scene for tile {tile_raw} up to {date}")

        item = items[0]
        tmp   = tempfile.mkdtemp(prefix=f"msi_{scene_id.replace('/','_')}_")
        files: Dict[str, str] = {}

        for logical, aliases in BANDS.items():
            asset = next((item.assets[a] for a in aliases if a in item.assets), None)
            if not asset:
                raise KeyError(f"{logical} missing, available={list(item.assets)}")
            ext  = ".jp2" if asset.href.lower().endswith(".jp2") else ".tif"
            path = os.path.join(tmp, f"{logical}{ext}")
            _download(asset.href, path, logical)
            files[logical] = path

        return {"scene_id": scene_id, "files": files}

    # ------------------------------------------------------------------ #
    def to_zarr(self, local_paths: Dict, zarr_path: str, **_) -> None:
        # Use B02 as reference grid (10 m)
        ref_path = local_paths["files"]["B02"]
        with rasterio.open(ref_path) as ref:
            ref_profile = ref.profile
            ref_arr     = ref.read(1)
            t           = ref.transform
            x = np.arange(ref.width)  * t.a + t.c
            y = np.arange(ref.height) * t.e + t.f
            coords = {"x": x, "y": y}

        data_arrays = [
            xr.DataArray(ref_arr, dims=("y", "x"), coords=coords, name="B02")
        ]

        for band, path in local_paths["files"].items():
            if band == "B02":
                continue
            with rasterio.open(path) as src:
                if (src.width, src.height) == (ref_profile["width"], ref_profile["height"]):
                    arr = src.read(1)
                else:
                    arr = _resample(path, ref_profile)
            data_arrays.append(xr.DataArray(arr, dims=("y", "x"), coords=coords, name=band))

        ds = xr.merge(data_arrays)
        ds.attrs["scene_id"] = local_paths["scene_id"]
        write_zarr(ds, zarr_path)
