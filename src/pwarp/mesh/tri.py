from __future__ import annotations

from pwarp import np
from pwarp.core import dtype

__all__ = ("triangular_mesh",)


def triangular_mesh(
        width: int,
        height: int,
        delta: int,
        method: str = "scipy",
) -> tuple[np.ndarray, np.ndarray]:
    """Create a triangular mesh inside a rectangular area.

    Build a 2D triangular mesh within a rectangle of size `width` by `height`.
    Control the mesh density with `delta`, which defines the spacing between
    vertices along the rectangle frame and the interior sampling grid.

    In the sketch below, `delta` is the distance from corner 0 to vertex x.

    ::

         0___x_______ width
        |
        x
        |
        |
        height

    Use `method="scipy"` to triangulate with SciPy's Delaunay implementation.
    Use `method="jrs"` to triangulate with the `triangle` package
    (Jonathan Richard Shewchuk).

    :param width: Rectangle width in pixels/units.
    :param height: Rectangle height in pixels/units.
    :param delta: Vertex spacing along the frame and sampling grid.
    :param method: Select the triangulation backend ("scipy" or "jrs").
    :return: Tuple of (vertices, faces) as int32 arrays.
    :raise ValueError: Raise when `delta` is not positive or method is unsupported.
    """
    if delta <= 0:
        msg = "delta must be positive"
        raise ValueError(msg)
    if width <= 0 or height <= 0:
        msg = "width and height must be positive"
        raise ValueError(msg)

    nx = width // delta + 1
    ny = height // delta + 1

    x = np.linspace(0, width, nx)
    y = np.linspace(0, height, ny)[1:]

    xs = np.concatenate([x, x[1:], np.zeros(len(y)), np.zeros(len(y) - 1) + width])
    ys = np.concatenate([np.zeros(len(x)), np.zeros(len(x) - 1) + height, y, y[:-1]])

    if method == "scipy":
        from scipy.spatial import Delaunay  # noqa: PLC0415

        x_coo, y_coo = np.meshgrid(
            np.arange(delta, width, delta),
            np.arange(delta, height, delta),
        )
        xs = np.concatenate((xs, x_coo.ravel()))
        ys = np.concatenate((ys, y_coo.ravel()))

        points = np.column_stack((xs, ys))
        triangulation = Delaunay(points)

        return (
            triangulation.points.astype(dtype.INT32),
            triangulation.simplices.astype(dtype.INT32),
        )

    if method == "jrs":
        import triangle  # noqa: PLC0415

        frame = np.array([xs, ys], dtype=dtype.INT32).T
        area = int(np.power(delta, 2) // 2)
        tri = triangle.triangulate({"vertices": frame}, f"a{area}q30")

        return (
            tri["vertices"].astype(dtype.INT32),
            tri["triangles"].astype(dtype.INT32),
        )

    msg = f"Unsupported method: {method!r}"
    raise ValueError(msg)
