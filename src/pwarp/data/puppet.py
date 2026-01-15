from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

    from pwarp.core import dtype

__all__ = (
    "PuppetObject",
)


@dataclass
class PuppetObject:
    vertices: dtype.FLOAT32
    faces: dtype.INT32
    no_faces: dtype.INT32
    no_vertices: dtype.INT32

    @property
    def r(self) -> np.ndarray | dtype.FLOAT32:
        return self.vertices

    @property
    def f(self) -> np.ndarray | dtype.FLOAT32:
        return self.faces
