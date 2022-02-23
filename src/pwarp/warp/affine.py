from typing import Tuple

from skimage._shared.utils import convert_to_float
from skimage.transform._warps_cy import _warp_fast

from pwarp import np
from pwarp.core import dtype


def pad(v: np.ndarray) -> np.ndarray:
    return np.hstack([v, np.ones((v.shape[0], 1))])


def unpad(v: np.ndarray) -> np.ndarray:
    return v[:, :-1]


def affine_transformation(src: np.ndarray, dst: np.ndarray):
    """
    Find definition of affine transformation defined by source -> destination points.
    :param src: np.ndarray;
    :param dst: np.ndarray;
    :return: np.ndarray;
    """
    # In order to solve the augmented matrix (incl translation),
    # it's required all vectors are augmented with a "1" at the end
    # -> Pad the features with ones, so that our transformation can do translations too

    # Pad to [[ x y 1] , [x y 1]]
    y, x = pad(dst), pad(src)

    # Solve the least squares problem X * A = Y
    # and find the affine transformation matrix A.
    a_matrix, res, rank, s = np.linalg.lstsq(x, y, rcond=None)
    # a_matrix[np.abs(a_matrix) < 1e-10] = 0  # set really small values to zero

    return a_matrix.T


def warp(
        image: np.ndarray,
        warp_matrix: np.ndarray,
        output_shape: Tuple = None,
        mode: str = 'constant',
        cval: float = 0.0
) -> np.ndarray:
    image = convert_to_float(image, preserve_range=True)
    input_shape = np.array(image.shape)

    if output_shape is None:
        output_shape = input_shape

    matrix = warp_matrix.astype(image.dtype)
    ctype = 'float32_t' if image.dtype == np.float32 else 'float64_t'

    warped = np.zeros((output_shape[0], output_shape[1], 3), dtype=dtype.UINT8)
    for dim in range(image.shape[2]):
        warped[:, :, dim] = _warp_fast[ctype](image[..., dim], matrix,
                                              output_shape=output_shape,
                                              order=1, mode=mode,
                                              cval=cval)
    return warped
