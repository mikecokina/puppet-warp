from typing import Union, Tuple

from pwarp import np
from pwarp.core import dtype


__all__ = (
    'triangular_mesh',
)


def triangular_mesh(
        width: Union[int, dtype.INT32],
        height: Union[int, dtype.INT32],
        delta: Union[int, dtype.INT32],
        method: str = 'scipy'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create triangular mesh within rectangular area defined by width/height.
    Density of triangular mesh is adjustable via parameter delta, which defines
    distance between two vertices situated on frame of rectangle.
    In sketch bellow, delta is distance from corner 0 to vertex x.

    ::

         0___x_______ width
        |
        x
        |
        |
        height

    :param width: Union[int, dtype.INT32];
    :param height: Union[int, dtype.INT32];
    :param delta: Union[int, dtype.INT32];
    :param method: str; method to create triangular mesh; `scipy` or `jrs` (stands for Jonathan Richard Shewchuk)
    :return: Tuple[np.ndarray, np.ndarray];
    """
    nx, ny = width // delta + 1, height // delta + 1
    x = np.linspace(0, width, nx)
    y = np.linspace(0, height, ny)[1:]

    xs = np.concatenate([x, x[1:], np.zeros(len(y)), np.zeros(len(y) - 1) + width])
    ys = np.concatenate([np.zeros(len(x)), np.zeros(len(x) - 1) + height, y, y[:-1]])

    if method == 'scipy':
        from scipy.spatial import Delaunay

        x_coo, y_coo = np.meshgrid(np.arange(delta, width, delta), np.arange(delta, height, delta))
        xs = np.concatenate((xs, x_coo.ravel()))
        ys = np.concatenate((ys, y_coo.ravel()))

        # Triangulate points using Delaunay
        points = np.column_stack((xs, ys))
        triangulation = Delaunay(points)

        return triangulation.points.astype(dtype.INT32), triangulation.simplices.astype(dtype.INT32)

    elif method == 'jrs':
        import triangle

        frame = np.array([xs, ys]).astype(dtype=dtype.INT32).T
        area = np.power(delta, 2) // 2
        t = triangle.triangulate({"vertices": frame}, f'a{area}q30')
        return t['vertices'].astype(dtype.INT32), t['triangles'].astype(dtype.INT32)



