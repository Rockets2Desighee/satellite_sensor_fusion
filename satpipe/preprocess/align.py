# satpipe/preprocess/align.py
"""Spatial alignment / resampling to a common 10 m grid."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import mlflow
import numpy as np
import rasterio as rio
import xarray as xr
from rasterio.warp import reproject, Resampling

__all__ = ["regrid_to_10m"]

logger = logging.getLogger(__name__)


def _target_grid(reference_path: Path) -> Dict:
    """Return (transform, width, height, crs) from a 10 m reference band."""
    with rio.open(reference_path) as ref:
        return {
            "transform": ref.transform,
            "width": ref.width,
            "height": ref.height,
            "crs": ref.crs,
        }


def _reproject_band(
    src_da: xr.DataArray,
    tgt_meta: Dict,
    resampling: Resampling = Resampling.bilinear,
) -> xr.DataArray:
    """Rasterio-based reprojection → returns an in-memory xarray.DataArray."""
    dst_arr = np.empty((tgt_meta["height"], tgt_meta["width"]), dtype=np.float32)
    reproject(
        src_da.data,
        dst_arr,
        src_transform=src_da.rio.transform(),
        src_crs=src_da.rio.crs,
        dst_transform=tgt_meta["transform"],
        dst_crs=tgt_meta["crs"],
        resampling=resampling,
        num_threads=4,
    )
    out = xr.DataArray(
        dst_arr,
        dims=("y", "x"),
        attrs=src_da.attrs | {"transform": tgt_meta["transform"], "crs": tgt_meta["crs"]},
    )
    return out


def regrid_to_10m(
    src_zarr: Path | str,
    dst_zarr: Path | str,
    reference_band: str = "B04",
    **mlflow_tags: str,
) -> Path:
    """
    Upsample **all** bands to the 10 m grid defined by `reference_band`.

    Parameters
    ----------
    src_zarr
        Normalised Zarr store (input of previous stage).
    dst_zarr
        Aligned output Zarr (`data/processed/aligned/*.zarr`).
    reference_band
        A 10 m band to derive the target geospatial grid (B02, B03, B04, B08).
    """
    src_zarr, dst_zarr = Path(src_zarr), Path(dst_zarr)
    with mlflow.start_run(run_name="align_regrid_10m"):
        mlflow.log_param("reference_band", reference_band)
        mlflow.set_tags(mlflow_tags)

        ds = xr.open_zarr(src_zarr, consolidated=True)
        ref_da = ds[reference_band]
        tgt_meta = _target_grid(ref_da.encoding["source"])

        aligned = xr.merge(
            {
                band: _reproject_band(da, tgt_meta, Resampling.bilinear)
                for band, da in ds.data_vars.items()
            },
            compat="override",
            combine_attrs="override",
        )
        aligned.to_zarr(dst_zarr, mode="w", consolidated=True)
        mlflow.log_metric("n_bands", len(ds.data_vars))
        mlflow.log_metric("out_shape_y", tgt_meta["height"])
        mlflow.log_metric("out_shape_x", tgt_meta["width"])
        ds.close()
        aligned.close()

    logger.info("Spatially aligned Zarr saved @ %s", dst_zarr)
    return dst_zarr
