from __future__ import annotations

from pwarp import np


def is_close(
        x: int,
        y: int,
        vertices: np.ndarray,
        _tol: int = 4,
) -> tuple[bool, int]:
    """Check whether a point is close to any vertex and return its index.

    Scan a square neighborhood around the given point and determine whether
    any vertex lies within the specified tolerance. If a match is found,
    return True and the index of the first matching vertex. Otherwise,
    return False and -1.

    :param x: X coordinate of the query point.
    :param y: Y coordinate of the query point.
    :param vertices: Array of vertex coordinates of shape (N, 2).
    :param _tol: Pixel tolerance used to search around the query point.
    :return: Tuple of (is_close, vertex_index).
    """
    vertices_scaled = vertices.copy().astype(int)
    close = False
    index = -1

    for i in range(x - _tol, x + _tol):
        for j in range(y - _tol, y + _tol):
            click = np.array([i, j])
            matches = np.all(vertices_scaled == click, axis=1)
            if matches.any():
                close = True
                index = int(np.where(matches)[0][0])
                break

    return close, index
