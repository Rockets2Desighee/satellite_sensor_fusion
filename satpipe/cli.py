# satpipe/cli.py

import os, click, mlflow
from satpipe.ingest.msi import MSIIngestor
from satpipe.ingest.sar import SARIngestor

@click.group()
def cli():
    pass

@cli.command()
@click.argument("scene_id")
@click.option("--sensor", type=click.Choice(["msi", "sar"]), required=True)
def ingest(scene_id, sensor):
    """Ingests satellite scene and stores as Zarr (DVC + MLflow ready)."""
    safe_id = scene_id.replace("/", "_")
    zarr_path = f"data/raw/{sensor}/{safe_id}.zarr"
    os.makedirs(os.path.dirname(zarr_path), exist_ok=True)

    mlflow.set_tracking_uri("file:./mlruns")
    with mlflow.start_run(run_name=f"{sensor}_ingest_{safe_id}"):
        mlflow.log_param("scene_id", scene_id)
        mlflow.log_param("sensor", sensor)
        mlflow.log_param("zarr_output", zarr_path)

        if sensor == "msi":
            MSIIngestor().ingest(scene_id, zarr_path=zarr_path)
        else:
            SARIngestor().ingest(scene_id, zarr_path=zarr_path)

        # mlflow.log_artifact(zarr_path)

if __name__ == "__main__":
    cli()
