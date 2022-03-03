from dataclasses import dataclass

import numpy as np

from pwarp.core import dtype

__all__ = (
    'PuppetObject',
)


@dataclass
class PuppetObject(object):
    vertices: dtype.FLOAT32
    faces: dtype.INT32
    no_faces: dtype.INT32
    no_vertices: dtype.INT32

    @property
    def r(self) -> np.ndarray:
        return self.vertices

    @property
    def f(self) -> np.ndarray:
        return self.faces
