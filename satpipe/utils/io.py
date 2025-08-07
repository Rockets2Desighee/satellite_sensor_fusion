# satpipe/utils/io.py

import hashlib
import xarray as xr

def compute_scene_hash(dataset: xr.Dataset) -> str:
    meta = dataset.attrs.get("scene_id", "") + str(dataset.dims)
    return hashlib.sha256(meta.encode()).hexdigest()

def write_zarr(dataset: xr.Dataset, zarr_path: str):
    dataset.attrs["hash"] = compute_scene_hash(dataset)
    dataset.to_zarr(zarr_path, mode="w", consolidated=True)
