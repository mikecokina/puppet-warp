from __future__ import annotations

from typing import TYPE_CHECKING

import cv2

if TYPE_CHECKING:
    from pwarp import np


def draw_mesh(
        vertices: np.ndarray,
        edges: np.ndarray,
        img: np.ndarray,
        color: tuple[int, int, int] = (0, 255, 0),
) -> None:
    """Draw mesh edges onto an image.

    Iterate over the edge list and draw a line for each edge using the
    corresponding vertex coordinates.

    :param vertices: Array of vertex coordinates of shape (N, 2).
    :param edges: Array of edge indices of shape (M, 2).
    :param img: Image array to draw into.
    :param color: BGR color used to draw mesh edges.
    """
    for edge in edges:
        start = (
            int(vertices[int(edge[0]), 0]),
            int(vertices[int(edge[0]), 1]),
        )
        end = (
            int(vertices[int(edge[1]), 0]),
            int(vertices[int(edge[1]), 1]),
        )
        cv2.line(img, start, end, color, 1)
