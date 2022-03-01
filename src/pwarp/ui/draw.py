from typing import Tuple

import cv2

from pwarp import np


def draw_mesh(
        vertices: np.ndarray,
        edges: np.ndarray,
        img: np.ndarray,
        color: Tuple = (0, 255, 0)
) -> None:
    for edge in edges:
        start = (vertices[int(edge[0]), 0], vertices[int(edge[0]), 1])
        end = (vertices[int(edge[1]), 0], vertices[int(edge[1]), 1])
        cv2.line(img, start, end, color, 1)
