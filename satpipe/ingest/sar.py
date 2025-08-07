# satpipe/ingest/sar.py

import os, requests, rasterio
import xarray as xr
import numpy as np
from satpipe.ingest.base import AbstractIngestor
from satpipe.utils.io import write_zarr

S1_BASE = "https://sentinel-s1-l1c.s3.amazonaws.com"

class SARIngestor(AbstractIngestor):
    def download(self, scene_id: str, **kwargs):
        path = f"{S1_BASE}/{scene_id}/measurement/{scene_id}-VH.tiff"
        r = requests.get(path)
        if r.status_code != 200:
            raise FileNotFoundError(f"Cannot fetch {path}")
        os.makedirs(f"/tmp/{scene_id}", exist_ok=True)
        local_file = f"/tmp/{scene_id}/VH.tif"
        with open(local_file, 'wb') as f:
            f.write(r.content)
        return {"scene_id": scene_id, "files": {"VH": local_file}}

    def to_zarr(self, local_paths, zarr_path, **kwargs):
        with rasterio.open(local_paths["files"]["VH"]) as src:
            arr = src.read(1)
            transform = src.transform
            x = np.arange(arr.shape[1]) * transform.a + transform.c
            y = np.arange(arr.shape[0]) * transform.e + transform.f
            ds = xr.Dataset({
                "VH": xr.DataArray(arr, dims=["y", "x"], coords={"x": x, "y": y})
            })
            ds.attrs["scene_id"] = local_paths["scene_id"]
            write_zarr(ds, zarr_path)
