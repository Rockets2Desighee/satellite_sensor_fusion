# satpipe/ingest/base.py

from abc import ABC, abstractmethod
from typing import Dict, Any

class AbstractIngestor(ABC):
    @abstractmethod
    def download(self, scene_id: str, **kwargs) -> Dict[str, Any]:
        """Download raw scene and return local paths + metadata."""
        pass

    @abstractmethod
    def to_zarr(self, local_paths: Dict[str, Any], zarr_path: str, **kwargs) -> None:
        """Convert scene to Zarr format."""
        pass

    def ingest(self, scene_id: str, zarr_path: str, **kwargs) -> None:
        """End-to-end ingestion pipeline."""
        local = self.download(scene_id, **kwargs)
        self.to_zarr(local, zarr_path, **kwargs)
