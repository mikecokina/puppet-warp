import cv2
import numpy as np


def draw_mesh(vertices, edges, img):
    # scaling so that it fits in the window for OpenCV coordinate system
    vertices_scaled = np.zeros(np.shape(vertices))
    vertices_scaled[:, 0] = vertices[:, 0] * -180 + 640
    vertices_scaled[:, 1] = vertices[:, 1] * -180 + 400
    vertices_scaled = vertices_scaled.astype(int)
    for edge in edges:
        start = (vertices_scaled[int(edge[0] - 1), 0], vertices_scaled[int(edge[0] - 1), 1])
        end = (vertices_scaled[int(edge[1] - 1), 0], vertices_scaled[int(edge[1] - 1), 1])
        cv2.line(img, start, end, (0, 255, 0), 1)
