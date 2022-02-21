from typing import Tuple

import cv2
import numpy as np


from pwarp.core import dtype


def tri_warp(
        src: np.ndarray,
        tri_src: np.ndarray,
        tri_dst: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """

    :param src: np.ndarray;
    :param tri_src: np.ndarray;
    :param tri_dst: np.ndarray;
    :return: Tuple[np.ndarray, np.ndarray]; (transformed image within dst triangle, mask)
    """
    # Find bounding rectangle for each triangle
    bbox_src = cv2.boundingRect(tri_src)
    bbox_dst = cv2.boundingRect(tri_dst)

    # Offset points by left top corner of the respective rectangles.
    tri_src_cropped = tri_src[0, :, :] - np.array(bbox_src[:2], dtype=dtype.FLOAT32).astype(dtype.INT32)
    tri_dst_cropped = tri_dst[0, :, :] - np.array(bbox_dst[:2], dtype=dtype.FLOAT32).astype(dtype.INT32)

    # Crop input image.
    src_cropped = src[bbox_src[1]:bbox_src[1] + bbox_src[3], bbox_src[0]:bbox_src[0] + bbox_src[2]]

    # Given a pair of triangles, find the affine transform.
    # Warning: Following idiot requires float32.
    fn = cv2.getAffineTransform
    warp_matrix = fn(dtype.FLOAT32(tri_src_cropped), dtype.FLOAT32(tri_dst_cropped)).astype(dtype.FLOAT32)

    # Apply the Affine Transform  to the src image.
    _kwargs = dict(flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    dst_cropped = cv2.warpAffine(src_cropped, warp_matrix, (bbox_dst[2], bbox_dst[3]), None, **_kwargs)

    # Get mask by filling triangle.
    mask = np.zeros((bbox_dst[3], bbox_dst[2], 3), dtype=dtype.UINT8)
    cv2.fillConvexPoly(mask, dtype.INT32(tri_dst_cropped), (1.0, 1.0, 1.0), 16, 0)
    dst_cropped *= mask

    origin_mask = np.zeros(src.shape[:2], dtype=dtype.BOOL)
    origin_mask[bbox_dst[1]:bbox_dst[1] + bbox_dst[3], bbox_dst[0]:bbox_dst[0] + bbox_dst[2]] = mask[:, :, 0]

    # Prepare destination array,
    # Output image is set to white
    dst = 255 * np.ones(src.shape, dtype=dtype.UINT8)

    # Copy triangular region of the rectangular patch to the output image.
    dst[bbox_dst[1]:bbox_dst[1] + bbox_dst[3], bbox_dst[0]:bbox_dst[0] + bbox_dst[2]] = \
        dst[bbox_dst[1]:bbox_dst[1] + bbox_dst[3], bbox_dst[0]:bbox_dst[0] + bbox_dst[2]] * ((1.0, 1.0, 1.0) - mask)

    dst[bbox_dst[1]:bbox_dst[1] + bbox_dst[3], bbox_dst[0]:bbox_dst[0] + bbox_dst[2]] = \
        dst[bbox_dst[1]:bbox_dst[1] + bbox_dst[3], bbox_dst[0]:bbox_dst[0] + bbox_dst[2]] + dst_cropped

    return dst, origin_mask


if __name__ == '__main__':
    # Read input image
    imgIn = cv2.imread("robot.jpg")

    # Input triangle
    triIn = np.array([[[360, 200], [60, 250], [450, 400]]])

    # Output triangle
    triOut = np.array([[[400, 200], [160, 270], [400, 400]]])

    # Warp all pixels inside input triangle to output triangle
    imgOut, _ = tri_warp(imgIn, triIn, triOut)

    # Draw triangle using this color
    color = (255, 150, 0)

    # Draw triangles in input and output images.
    cv2.polylines(imgIn, triIn.astype(int), True, color, 2, 16)
    cv2.polylines(imgOut, triOut.astype(int), True, color, 2, 16)

    cv2.imshow("Input", imgIn)
    cv2.imshow("Output", imgOut)

    cv2.waitKey(0)
