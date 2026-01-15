from __future__ import annotations

# noinspection PyProtectedMember
from skimage._shared.utils import convert_to_float

# noinspection PyProtectedMember
from skimage.transform._warps_cy import _warp_fast

from pwarp import np
from pwarp.core import dtype


def pad(v: np.ndarray) -> np.ndarray:
    """Pad coordinates with a homogeneous 1 column.

    Append a column of ones to an (N, 2) array to form homogeneous coordinates
    suitable for affine transforms with translation.

    :param v: Coordinate array of shape (N, 2).
    :return: Homogeneous coordinate array of shape (N, 3).
    """
    return np.hstack([v, np.ones((v.shape[0], 1), dtype=v.dtype)])


def unpad(v: np.ndarray) -> np.ndarray:
    """Drop the homogeneous coordinate column.

    Remove the last column from a homogeneous coordinate array.

    :param v: Homogeneous coordinate array of shape (N, 3).
    :return: Coordinate array of shape (N, 2).
    """
    return v[:, :-1]


def affine_transformation(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Compute an affine transform matrix that maps src points to dst points.

    Solve a least-squares problem to find the affine transformation in
    homogeneous coordinates. The returned matrix is 3x3.

    :param src: Source points of shape (N, 2).
    :param dst: Destination points of shape (N, 2).
    :return: Affine transform matrix of shape (3, 3).
    """
    y_mat = pad(dst)
    x_mat = pad(src)

    a_matrix, _res, _rank, _s = np.linalg.lstsq(x_mat, y_mat, rcond=None)
    return a_matrix.T


def warp(
        image: np.ndarray,
        warp_matrix: np.ndarray,
        output_shape: tuple[int, int] | None = None,
        mode: str = "constant",
        cval: float = 0.0,
) -> np.ndarray:
    """Warp an image using an affine transform.

    Apply the given affine transform matrix to an image using scikit-image's
    fast warping kernel. Produce an output image with the requested shape.

    :param image: Input image array of shape (H, W, C).
    :param warp_matrix: Affine transform matrix of shape (3, 3).
    :param output_shape: Output shape as (H, W). Use input shape when None.
    :param mode: Border mode passed to the warping kernel.
    :param cval: Constant value used when mode is "constant".
    :return: Warped image as uint8 array of shape (H, W, C).
    """
    image_f = convert_to_float(image, preserve_range=True)

    if output_shape is None:
        output_shape = (int(image_f.shape[0]), int(image_f.shape[1]))
    else:
        output_shape = (int(output_shape[0]), int(output_shape[1]))

    matrix = warp_matrix.astype(image_f.dtype, copy=False)
    ctype = "float32_t" if image_f.dtype == np.float32 else "float64_t"

    warped = np.zeros((output_shape[0], output_shape[1], int(image_f.shape[2])), dtype=dtype.UINT8)
    for dim in range(int(image_f.shape[2])):
        # noinspection PyUnresolvedReferences
        warped[:, :, dim] = _warp_fast[ctype](
            image_f[..., dim],
            matrix,
            output_shape=output_shape,
            order=1,
            mode=mode,
            cval=cval,
        )

    return warped
