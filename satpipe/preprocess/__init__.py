# satpipe/preprocess/__init__.py
"""
satpipe.preprocess
==================

Pre-processing pipeline components:
-----------------------------------
* :pymod:`satpipe.preprocess.normalize`  – per-band statistical normalisation
* :pymod:`satpipe.preprocess.align`      – spatial re-gridding to a common 10 m EO grid
"""

from .normalize import Normaliser, compute_stats, normalise_zarr  # noqa: F401
from .align import regrid_to_10m  # noqa: F401
