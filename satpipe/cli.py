# # satpipe/cli.py

# import os, click, mlflow
# from satpipe.ingest.msi import MSIIngestor
# from satpipe.ingest.sar import SARIngestor

# @click.group()
# def cli():
#     pass

# @cli.command()
# @click.argument("scene_id")
# @click.option("--sensor", type=click.Choice(["msi", "sar"]), required=True)
# def ingest(scene_id, sensor):
#     """Ingests satellite scene and stores as Zarr (DVC + MLflow ready)."""
#     safe_id = scene_id.replace("/", "_")
#     zarr_path = f"data/raw/{sensor}/{safe_id}.zarr"
#     os.makedirs(os.path.dirname(zarr_path), exist_ok=True)

#     mlflow.set_tracking_uri("file:./mlruns")
#     with mlflow.start_run(run_name=f"{sensor}_ingest_{safe_id}"):
#         mlflow.log_param("scene_id", scene_id)
#         mlflow.log_param("sensor", sensor)
#         mlflow.log_param("zarr_output", zarr_path)

#         if sensor == "msi":
#             MSIIngestor().ingest(scene_id, zarr_path=zarr_path)
#         else:
#             SARIngestor().ingest(scene_id, zarr_path=zarr_path)

#         # mlflow.log_artifact(zarr_path)

# if __name__ == "__main__":
#     cli()


# satpipe/cli.py
"""Project-wide CLI entry-points.

Usage examples
--------------
# Ingest a Sentinel-2 MSI scene
satpipe ingest  S2A_MSIL2A_20250715T051641_N0514_R019_T44RFQ --sensor msi

# Normalise the raw Zarr produced above
satpipe normalise  data/raw/msi/S2A_MSIL2A_20250715T051641_N0514_R019_T44RFQ.zarr \
                   data/processed/S2A_MSIL2A_20250715_norm.zarr  --mode zscore

# Align everything to a 10 m grid
satpipe align      data/processed/S2A_MSIL2A_20250715_norm.zarr \
                   data/processed/S2A_MSIL2A_20250715_align.zarr  --reference-band B04
"""

from __future__ import annotations

import os
from pathlib import Path

import click
import mlflow

from satpipe.ingest.msi import MSIIngestor
from satpipe.ingest.sar import SARIngestor
from satpipe.preprocess import normalise_zarr, regrid_to_10m


@click.group()
def cli() -> None:  # noqa: D401
    """Top-level command group for *satpipe*."""
    pass


# --------------------------------------------------------------------------- #
#                               INGEST COMMAND                                #
# --------------------------------------------------------------------------- #
@cli.command()
@click.argument("scene_id")
@click.option("--sensor", type=click.Choice(["msi", "sar"]), required=True)
def ingest(scene_id: str, sensor: str) -> None:
    """Ingest a scene and store it as Zarr (tracked by DVC, logged in MLflow)."""
    safe_id = scene_id.replace("/", "_")
    zarr_path = f"data/raw/{sensor}/{safe_id}.zarr"
    os.makedirs(os.path.dirname(zarr_path), exist_ok=True)

    mlflow.set_tracking_uri("file:./mlruns")
    with mlflow.start_run(run_name=f"{sensor}_ingest_{safe_id}"):
        mlflow.log_params({"scene_id": scene_id, "sensor": sensor, "zarr_output": zarr_path})

        if sensor == "msi":
            MSIIngestor().ingest(scene_id, zarr_path=zarr_path)
        else:
            SARIngestor().ingest(scene_id, zarr_path=zarr_path)


# --------------------------------------------------------------------------- #
#                            PRE-PROCESSING COMMANDS                          #
# --------------------------------------------------------------------------- #
@cli.command()
@click.argument("src", type=click.Path(exists=True, path_type=Path))
@click.argument("dst", type=click.Path(path_type=Path))
@click.option("--mode", type=click.Choice(["zscore", "minmax"]), default="zscore")
def normalise(src: Path, dst: Path, mode: str) -> None:
    """Per-band statistical normalisation."""
    mlflow.set_tracking_uri("file:./mlruns")
    normalise_zarr(src, dst, mode=mode)  # internal function handles MLflow run


@cli.command()
@click.argument("src", type=click.Path(exists=True, path_type=Path))
@click.argument("dst", type=click.Path(path_type=Path))
@click.option("--reference-band", default="B04", show_default=True)
def align(src: Path, dst: Path, reference_band: str) -> None:
    """Spatially re-grid *all* bands to the 10 m grid of `reference_band`."""
    mlflow.set_tracking_uri("file:./mlruns")
    regrid_to_10m(src, dst, reference_band=reference_band)  # handles MLflow run


if __name__ == "__main__":
    cli()
