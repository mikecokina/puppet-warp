from typing import Union, Iterable, Tuple

import cv2
import numpy as np

from pwarp.core import dtype
from pwarp.warp import affine
from pwarp.profiler import profileit

__all__ = (
    'graph_defined_warp',
)


def _broadcast_transformed_tri(
        dst: np.ndarray,
        bbox: Tuple[int, int, int, int],
        warped: np.ndarray,
        mask: np.ndarray
) -> np.ndarray:
    """

    :param dst:
    :param bbox:
    :param warped:
    :param mask:
    :return:
    """
    # Copy triangular region of the rectangular patch to the output image.
    dst[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]] = \
        dst[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]] * ((1.0, 1.0, 1.0) - mask)

    dst[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]] = \
        dst[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]] + warped

    return dst


def inbox_tri_warp(
        src: np.ndarray,
        tri_src: np.ndarray,
        tri_dst: np.ndarray,
        use_scikit: bool = True
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int]]:
    """
    Based on src triangle and dst trianglle in 2D will find affine transformation and return this
    transformation in bounding box of destination triangle.

    :param src: np.ndarray;
    :param tri_src: np.ndarray;
    :param tri_dst: np.ndarray;
    :param use_scikit: bool;
    :return: Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int]];
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
    if use_scikit:
        # Scikit based warp requires inverse approach to source/destination points.
        scikit_warp_matrix = affine.affine_transformation(tri_dst_cropped, tri_src_cropped)
        dst_cropped = affine.warp(src_cropped, scikit_warp_matrix, mode='edge', output_shape=(bbox_dst[3], bbox_dst[2]))
    else:
        cv2_warp_matrix = affine.affine_transformation(tri_src_cropped, tri_dst_cropped)[:2, :]
        # Apply the Affine Transform  to the src image.
        _kwargs = dict(flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
        dst_cropped = cv2.warpAffine(src_cropped, cv2_warp_matrix, (bbox_dst[2], bbox_dst[3]), None, **_kwargs)

    # Get mask by filling triangle.
    mask = np.zeros((bbox_dst[3], bbox_dst[2], 3), dtype=dtype.UINT8)
    cv2.fillConvexPoly(mask, dtype.INT32(tri_dst_cropped), (1, 1, 1), 16, 0)
    dst_cropped *= mask

    return dst_cropped, mask, bbox_dst


def tri_warp(
        src: np.ndarray,
        tri_src: np.ndarray,
        tri_dst: np.ndarray,
        use_scikit: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Based on src triangle and dst trianglle in 2D will find affine transformation
    and returns transformed image from src image in area of dst triangle as well
    as boolean mask which defines this triangle location.

    :param src: np.ndarray;
    :param tri_src: np.ndarray;
    :param tri_dst: np.ndarray;
    :param use_scikit: bool;
    :return: np.ndarray; (transformed image within triangle, boolean mask for triangle visibility only)
    """
    dst_cropped, mask, bbox_dst = inbox_tri_warp(src, tri_src, tri_dst, use_scikit)
    alpha = np.zeros(src.shape[:2], dtype=dtype.BOOL)
    alpha[bbox_dst[1]:bbox_dst[1] + bbox_dst[3], bbox_dst[0]:bbox_dst[0] + bbox_dst[2]] = mask[:, :, 0]

    # Prepare destination array,
    # Output image is set to white
    dst = np.ones(src.shape, dtype=dtype.UINT8) * 255
    # Broadcast pixels within base image.
    dst = _broadcast_transformed_tri(dst, bbox_dst, dst_cropped, mask)
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


def graph_defined_warp(
        image: np.ndarray,
        vertices_src: np.ndarray,
        faces_src: np.ndarray,
        vertices_dst: np.ndarray,
        faces_dst: np.ndarray,
        use_scikit: Union[dtype.BOOL, bool] = False,
) -> np.ndarray:
    """
    Based on triangulated shape transformed from source to destination mesh
    will provide transformation of image. The idea is to do an affine transformation
    of each triangle in mesh. The affine transformation is definde for each triangle
    separately and partial results are merged at the end of the process.

    :param image: np.ndarray; image
    :param vertices_src: np.ndarray;
    :param faces_src: np.ndarray;
    :param vertices_dst: np.ndarray;
    :param faces_dst: np.ndarray;
    :param use_scikit: bool;
    :return: np.ndarray;
    """
    # Create base white image.
    base_image = np.ones(image.shape, dtype=dtype.UINT8) * 255

    # Iterate over all faces.
    for f_src, f_dst in zip(faces_src, faces_dst):
        # Choose corresponding vertices defined by faces.
        r_src, r_dst = vertices_src[f_src], vertices_dst[f_dst]
        r_src, r_dst = np.array([r_src], dtype=r_src.dtype), np.array([r_dst], dtype=r_dst.dtype)
        # Transform given triangles pair in affine manner.
        warped, alpha, bbox = inbox_tri_warp(image, r_src, r_dst, use_scikit=use_scikit)

        # Put transformed data within base image based on alpha mask and bounding box of given destination face.
        # Copy triangular region of the rectangular patch to the output image.
        base_image = _broadcast_transformed_tri(base_image, bbox, warped, alpha)

    return base_image
