"""Optional CPU reconstruction producer; importing this package loads no models.

Install the ``world-reconstruct`` extra and supply GLOMAP/DA3 runtime artifacts
only when running the producer. Reading an existing run uses the standard-library
``lingshu.world.reconstruction`` module instead.
"""

from __future__ import annotations

__all__ = ["ReconstructionPipeline"]


def __getattr__(name: str):
    if name == "ReconstructionPipeline":
        from .pipeline import ReconstructionPipeline
        return ReconstructionPipeline
    raise AttributeError(name)
