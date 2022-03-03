from typing import Union, Iterable, Tuple

import cv2
import numpy as np

from pwarp.core import dtype, ops
from pwarp.core.arap import StepOne, StepTwo
from pwarp.warp import affine

__all__ = (
    'graph_defined_warp',
    'graph_warp',
)


def _broadcast_transformed_tri(
        dst: np.ndarray,
        bbox: Tuple[int, int, int, int],
        warped: np.ndarray,
        mask: np.ndarray
) -> np.ndarray:
    """
    Broadcast triangle transformed within bounding box in affine manner
    into destination image of shape of original image.

    :param dst: np.ndarray;
    :param bbox: Tuple[int, int, int, int]; (top_left_x, top_left_y, width, height)
    :param warped: np.ndarray;
    :param mask: np.ndarray;
    :return: np.ndarray;
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


def graph_warp(
        vertices: np.ndarray,
        faces: np.ndarray,
        control_indices: np.ndarray,
        shifted_locations: np.ndarray,
        edges: np.ndarray = None,
        precomputed: Tuple[np.ndarray, np.ndarray, np.ndarray] = None
) -> np.ndarray:
    """
    Transform in ARAP manner graph defined by faces and vertices. The given graph will be transformed
    based on location of defined by shifted_locations. New location defined by shifted_locations
    correspond to original vertices via variable control_indices.

    Example::

        control_indices = np.array([22, 50, 94, 106], dtype=int)
        shifted_locations = np.array(
            [[0.555, -0.905],
             [-0.965, -0.875],
             [-0.950, 0.460],
             [0.705, 0.285]], dtype=float
        )

    :param vertices: np.ndarray; vertices
    :param faces: np.ndarray; faces
    :param control_indices: np.ndarray; indices of vertices in graph selected for transformation
    :param shifted_locations: np.ndarray; new location of control vertices
    :param edges: np.ndarray; set of edges defined by faces or None
    :param precomputed: Tuple[np.ndarray, np.ndarray, np.ndarray]; [gi, g_product, h] or None
    :return: np.ndarray; transformed vertices
    """
    if edges is None:
        edges = ops.get_edges(len(faces), faces)

    if precomputed is None:
        # Compute transformation matrices in case when not supplied.
        gi, g_product = StepOne.compute_g_matrix(vertices, edges, faces)
        h = StepOne.compute_h_matrix(edges, g_product, gi, vertices)
    else:
        gi, g_product, h = precomputed

    # Compute v' from paper.
    args = edges, vertices, gi, h, control_indices, shifted_locations
    new_vertices, _, _ = StepOne.compute_v_prime(*args)

    # Compute v'' from paper.
    t_matrix = StepTwo.compute_t_matrix(edges, g_product, gi, new_vertices)
    new_vertices = StepTwo.compute_v_2prime(edges, vertices, t_matrix, control_indices, shifted_locations)

    return new_vertices


# if __name__ == '__main__':
#     from pwarp import get_default_puppet
#     from matplotlib import pyplot as plt
#
#     control_pts = np.array([22, 50, 94, 106], dtype=int)
#     shift = np.array(
#         [[0.555, - 0.905],
#          [-0.965, - 0.875],
#          [-0.950, 0.460],
#          [0.705, 0.285]], dtype=float
#     )
#     puppet = get_default_puppet()
#     new_vertices = graph_warp(
#         vertices=puppet.r,
#         faces=puppet.f,
#         control_indices=control_pts,
#         shifted_locations=shift
#     )
#
#     fig, ax = plt.subplots(1, 2, frameon=False)
#     plt.tight_layout(pad=0)
#
#     ax[0].triplot(puppet.r.T[0], puppet.r.T[1], puppet.f)
#     ax[1].triplot(new_vertices.T[0], new_vertices.T[1], puppet.f)
#
#     for _ax in ax:
#         _ax.set_aspect('equal')
#         _ax.set_xlim([-1.2, 1.2])
#         _ax.set_ylim([-1.2, 1.2])
#         _ax.axis('off')
#     # plt.savefig("graph_transform.png", format='png', dpi=100)
#     plt.show()

#     from pwarp import _io
#     from pwarp.core import ops
#     from pwarp.ui import draw
#     from time import time
#
#     # Read input image
#     imgIn = cv2.imread("puppet.png")
#
#     _nr_src, _nf_src, _r_src, _f_src = _io.read_wavefront('../data/puppet.obj')
#     _edges_src = ops.get_edges(_nf_src, _f_src)
#     _r_src = draw.shift_scale(_r_src, dx=300, dy=440, scale=-290).astype(np.int32)
#
#     # draw.draw_mesh(_r_src, _edges_src, imgIn, color=(0, 0, 255))
#
#     _nr_dst, _nf_dst, _r_dst, _f_dst = _io.read_wavefront('../data/puppet_000.obj')
#     _edges_dst = ops.get_edges(_nf_dst, _f_dst)
#     _r_dst = draw.shift_scale(_r_dst, dx=300, dy=440, scale=-290).astype(np.int32)
#     # draw.draw_mesh(_r_dst, _edges_dst, imgIn, color=(255, 0, 0))
#
#     t = time()
#     graph_defined_warp(imgIn, vertices_src=_r_src, faces_src=_f_src, vertices_dst=_r_dst, faces_dst=_f_dst)
#     print(time() - t)
#
#     exit()
#
#     t = time()
#     parts = []
#
#
#     def do_tri():
#         for _ in range(10):
#             for _src_f, _dst_f in zip(_f_src, _f_dst):
#                 _src_r, _dst_r = _r_src[_src_f], _r_dst[_dst_f]
#                 _src_r, _dst_r = np.expand_dims(_src_r, axis=0), np.expand_dims(_dst_r, axis=0)
#                 img = tri_warp(imgIn, _src_r, _dst_r, use_scikit=False)
#                 parts.append(img)
#
#
#     do_tri()
#     print(time() - t)
#
#     h, w = imgIn.shape[:2]
#
#     t = time()
#
#
#     def merge():
#         imgOut = merge_transformed(parts, w, h)
#         return imgOut
#
#
#     imgOut = merge()
#     print(time() - t)
#     # exit()
#     # # Warp all pixels inside input triangle to output triangle
#     # imgOut, _ = tri_warp(imgIn, triIn, triOut)
#     #
#     # # Draw triangle using this color
#     # color = (255, 150, 0)
#     #
#     # cv2.polylines(imgIn, triIn.astype(int), True, color, 2, 16)
#     # cv2.polylines(imgOut, triOut.astype(int), True, color, 2, 16)
#
#     # # Draw triangles in input and output images.
#     # cv2.polylines(imgIn, triIn.astype(int), True, color, 2, 16)
#     # cv2.polylines(imgOut, triOut.astype(int), True, color, 2, 16)
#
#     cv2.imshow("Input", imgIn)
#     cv2.imshow("Output", imgOut)
#
#     cv2.waitKey(0)
