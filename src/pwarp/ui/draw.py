from typing import Union, Tuple

import cv2

from pwarp import np
from pwarp.core import dtype


def shift_scale(
        vertices: np.array,
        dx: Union[dtype.INT, int],
        dy: Union[dtype.INT, int],
        scale: Union[dtype.INT, int]
) -> np.ndarray:
    vertices = vertices.copy()
    vertices[:, 0] = vertices[:, 0] * scale + dx
    vertices[:, 1] = vertices[:, 1] * scale + dy
    return vertices


def draw_mesh(
        vertices: np.ndarray,
        edges: np.ndarray,
        img: np.ndarray,
        dx: Union[dtype.INT, int] = dtype.INT(0),
        dy: Union[dtype.INT, int] = dtype.INT(0),
        scale: Union[dtype.INT, int] = dtype.INT(1),
        color: Tuple = (0, 255, 0)
) -> None:
    # Scaling so that it fits in the window for OpenCV coordinate system.
    vertices_scaled = vertices.copy()
    if scale != 1 or dx != 0 or dy != 0:
        vertices_scaled = shift_scale(vertices, dx, dy, scale).astype(int)
    for edge in edges:
        start = (vertices_scaled[int(edge[0]), 0], vertices_scaled[int(edge[0]), 1])
        end = (vertices_scaled[int(edge[1]), 0], vertices_scaled[int(edge[1]), 1])
        cv2.line(img, start, end, color, 1)
