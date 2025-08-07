# satpipe/preprocess/normalize.py
"""Per-band normalisation utilities for Sentinel-2 MSI Zarr stores."""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import Dict, Literal, Tuple

import mlflow
import numpy as np
import xarray as xr

__all__ = ["Normaliser", "compute_stats", "normalise_zarr"]

logger = logging.getLogger(__name__)
_TStat = Tuple[float, float]  # (mean, std)


class Normaliser:
    """Statistical (z-score or min-max) normaliser for Sentinel-2 bands."""

    def __init__(
        self,
        stats: Dict[str, _TStat],
        mode: Literal["zscore", "minmax"] = "zscore",
        clip_sigma: float | None = 3.0,
    ) -> None:
        """
        Parameters
        ----------
        stats
            Mapping ``band_name → (mean, std | range)``.
        mode
            ``"zscore"`` or ``"minmax"`` normalisation.
        clip_sigma
            Optionally clip extreme values (``None`` disables).
        """
        self.stats = stats
        self.mode = mode
        self.clip_sigma = clip_sigma

    def _transform_band(self, arr: xr.DataArray, band: str) -> xr.DataArray:
        mean, spread = self.stats[band]
        if self.mode == "zscore":
            out = (arr - mean) / spread
            if self.clip_sigma is not None:
                out = out.clip(min=-self.clip_sigma, max=self.clip_sigma)
        else:  # minmax
            out = (arr - mean) / spread  # spread == max-min
            out = out.clip(0.0, 1.0)
        return out.astype(np.float32)

    def __call__(self, ds: xr.Dataset) -> xr.Dataset:
        """Return a *new* ``xarray.Dataset`` with all bands normalised."""
        return xr.merge(
            {b: self._transform_band(ds[b], b) for b in ds.data_vars},
            compat="override",
            combine_attrs="override",
        )


# --------------------------------------------------------------------------- #
#                PUBLIC API functions – pipeline-friendly helpers             #
# --------------------------------------------------------------------------- #


def compute_stats(
    zarr_path: Path | str,
    sample_frac: float = 0.02,
    rng_seed: int = 42,
) -> Dict[str, _TStat]:
    """
    Compute per-band mean & std **or** (min, range) on a random sample of pixels.

    Parameters
    ----------
    zarr_path
        Location of **raw** Sentinel-2 Zarr store (`data/raw/msi/...`).
    sample_frac
        Fraction of pixels to sample (default ≈ 2 % more than enough).
    rng_seed
        For reproducibility.

    Returns
    -------
    Dict[str, tuple[float, float]]
        ``band → (mean, std)`` suitable for :class:`Normaliser`.
    """
    rng = np.random.default_rng(rng_seed)
    ds = xr.open_zarr(zarr_path, consolidated=True)

    stats: Dict[str, _TStat] = {}
    for band, da in ds.data_vars.items():
        flat = da.data.reshape(-1)
        n = max(1, int(sample_frac * flat.size))
        sample = rng.choice(flat, size=n, replace=False)
        stats[band] = (float(sample.mean()), float(sample.std()))
        logger.debug("Stats[%s] – μ=%.3f σ=%.3f (n=%d)", band, *stats[band], n)

    ds.close()
    return stats


def normalise_zarr(
    src: Path | str,
    dst: Path | str,
    stats: Dict[str, _TStat] | None = None,
    mode: Literal["zscore", "minmax"] = "zscore",
    **mlflow_tags: str,
) -> Path:
    """
    End-to-end normalisation: read raw Zarr → write processed Zarr.

    Heavy outputs **must be DVC-tracked**; MLflow logs only params/metrics.

    Parameters
    ----------
    src
        Raw Zarr store (`data/raw/...`).
    dst
        Destination directory (`data/processed/...`). File is overwritten.
    stats
        Pre-computed stats (``None`` → compute on-the-fly on `src`).
    mode
        Normalisation strategy.
    **mlflow_tags
        Extra run tags (e.g. commit SHA, dataset version).
    """
    src = Path(src)
    dst = Path(dst)
    os.makedirs(dst.parent, exist_ok=True)

    with mlflow.start_run(run_name="normalise_zarr"):
        mlflow.log_param("mode", mode)
        mlflow.set_tags(mlflow_tags)

        if stats is None:
            stats = compute_stats(src)
            mlflow.log_param("stats_computed", True)
        else:
            mlflow.log_param("stats_computed", False)

        # Log a few descriptive stats – keeps MLflow UI informative but light.
        for b, (mu, sigma) in stats.items():
            mlflow.log_metric(f"{b}_mean_raw", mu)
            mlflow.log_metric(f"{b}_std_raw", sigma)

        norm = Normaliser(stats, mode=mode)
        logger.info("Loading raw Zarr %s …", src)
        ds = xr.open_zarr(src, consolidated=True, chunks="auto")
        logger.info("Normalising …")
        out = norm(ds)

        logger.info("Writing processed Zarr → %s", dst)
        out.to_zarr(dst, mode="w", consolidated=True)
        ds.close()
        out.close()

        # Verify numeric integrity & log quick sanity metric.
        example_band = next(iter(stats))
        max_abs = float(np.abs(xr.open_zarr(dst)[example_band]).max())
        mlflow.log_metric("max_abs_after_norm", max_abs)

    logger.info("Normalised dataset stored at %s (track with DVC)", dst)
    return dst
