from typing import Union, List, Iterable, Tuple

import cv2
import numpy as np

from pwarp.core import dtype


def tri_warp(
        src: np.ndarray,
        tri_src: np.ndarray,
        tri_dst: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Based on src triangle and dst trianglle in 2D will find affine transformation
    and returns transformed image from src image in area of dst triangle as well
    as boolean mask which defines this triangle location.

    :param src: np.ndarray;
    :param tri_src: np.ndarray;
    :param tri_dst: np.ndarray;
    :return: np.ndarray; (transformed image within triangle, boolean mask for triangle visibility only)
    """
    # Find bounding rectangle for each triangle
    bbox_src = cv2.boundingRect(tri_src)
    bbox_dst = cv2.boundingRect(tri_dst)

    # Offset points by left top corner of the respective rectangles.
    tri_src_cropped = tri_src[0, :, :] - bbox_src[:2]
    tri_dst_cropped = tri_dst[0, :, :] - bbox_dst[:2]

    # Crop input image.
    src_cropped = src[bbox_src[1]:bbox_src[1] + bbox_src[3], bbox_src[0]:bbox_src[0] + bbox_src[2]]

    # Given a pair of triangles, find the affine transform.
    # Warning: Following idiot requires float32.
    fn = cv2.getAffineTransform
    warp_matrix = fn(np.array(tri_src_cropped, dtype.FLOAT32), np.array(tri_dst_cropped, dtype.FLOAT32))

    # Apply the Affine Transform  to the src image.
    _kwargs = dict(flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    dst_cropped = cv2.warpAffine(src_cropped, warp_matrix, (bbox_dst[2], bbox_dst[3]), None, **_kwargs)

    # Get mask by filling triangle.
    mask = np.zeros((bbox_dst[3], bbox_dst[2], 3), dtype=dtype.UINT8)
    cv2.fillConvexPoly(mask, dtype.INT32(tri_dst_cropped), (1, 1, 1), 16, 0)
    dst_cropped *= mask

    alpha = np.zeros(src.shape[:2], dtype=dtype.BOOL)
    alpha[bbox_dst[1]:bbox_dst[1] + bbox_dst[3], bbox_dst[0]:bbox_dst[0] + bbox_dst[2]] = mask[:, :, 0]

    # Prepare destination array,
    # Output image is set to white
    dst = np.ones(src.shape, dtype=dtype.UINT8) * 255

    # Copy triangular region of the rectangular patch to the output image.
    dst[bbox_dst[1]:bbox_dst[1] + bbox_dst[3], bbox_dst[0]:bbox_dst[0] + bbox_dst[2]] = \
        dst[bbox_dst[1]:bbox_dst[1] + bbox_dst[3], bbox_dst[0]:bbox_dst[0] + bbox_dst[2]] * ((1.0, 1.0, 1.0) - mask)

    dst[bbox_dst[1]:bbox_dst[1] + bbox_dst[3], bbox_dst[0]:bbox_dst[0] + bbox_dst[2]] = \
        dst[bbox_dst[1]:bbox_dst[1] + bbox_dst[3], bbox_dst[0]:bbox_dst[0] + bbox_dst[2]] + dst_cropped

    return dst, alpha


def merge_transformed(
        parts: Iterable[Tuple[np.ndarray, np.ndarray]],
        width: Union[dtype.INT, int],
        height: Union[dtype.INT, int],
        channels: Union[dtype.INT, int] = 3,
        base_image: np.ndarray = None
) -> np.ndarray:
    """
    Merge together pieces of images defined by image and its mask.
    Mask is expected to be a boolean type np.ndarray and image 3 channels
    cv2 like image.

    :param parts: Iterable[Tuple[np.ndarray, np.ndarray]];
    :param width: int;
    :param height: int;
    :param channels: int;
    :param base_image: np.ndarray;
    :return: np.ndarray;
    """
    if base_image is None:
        base_image = np.ones((height, width, channels), dtype=np.uint8) * 255
    for image, alpha in parts:
        base_image[alpha, :] = image[alpha, :]
    return base_image
