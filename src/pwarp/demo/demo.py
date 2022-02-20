import sys

import cv2

from pwarp import np, _io
from pwarp.core import ops
from pwarp.core.arap import StepOne
from pwarp.demo import misc
from pwarp.settings import settings


class Demo(object):
    def __init__(self, obj_path: str, window_name: str = 'ARAP'):
        self.start_select = False
        self.end_select = False
        self.start_move = False
        self.end_move = False
        self.moving_index = -1
        self.selected_old = [-1]
        self.a_matrix = np.nan
        self.b_vector = np.nan
        self.moving_index_old = -1
        self.started = False
        self.new_vertices = np.nan

        self.window_name = window_name

        # Load wavefront object.
        num_vertices, num_faces, vertices, faces = _io.read_wavefront(obj_path)
        self.num_vertices = num_vertices
        self.num_faces = num_faces
        self.vertices = vertices
        self.faces = faces
        self.edges = ops.get_edges(num_faces, faces)

        self.vertices_move = np.zeros(self.num_vertices)
        self.vertices_select = np.zeros(self.num_vertices)

        self.img = np.zeros((800, 1280, 3), np.uint8)
        self._img = self.img.copy()

        # Compute initial transformation matrices
        self.gi, self.g_product = StepOne.compute_g_matrix(self.vertices, self.edges, self.faces)
        self.h = StepOne.compute_h_matrix(self.edges, self.g_product, self.gi, self.vertices)

    @classmethod
    def draw_mesh(cls, vertices, edges, img):
        # Scaling so that it fits in the window for OpenCV coordinate system.
        vertices_scaled = cls.inscreen_scale(vertices).astype(int)
        for edge in edges:
            start = (vertices_scaled[int(edge[0]), 0], vertices_scaled[int(edge[0]), 1])
            end = (vertices_scaled[int(edge[1]), 0], vertices_scaled[int(edge[1]), 1])
            cv2.line(img, start, end, (0, 255, 0), 1)

    @staticmethod
    def inscreen_scale(vertices: np.array):
        vertices = vertices.copy()
        vertices[:, 0] = vertices[:, 0] * -180 + 640
        vertices[:, 1] = vertices[:, 1] * -180 + 400
        return vertices

    @staticmethod
    def mouse_event_callback(event, *args):
        x, y, flags, param = args
        self: Demo = param

        if event == cv2.EVENT_LBUTTONDOWN:
            is_close, index = misc.is_close(x, y, self.vertices)

            if self.started:
                is_close, index = misc.is_close(x, y, self.new_vertices)

            if is_close:
                # Mark vertex as control point.
                if self.vertices_select[index] == 0:
                    cv2.circle(self.img, (x, y), 5, (0, 0, 255), 1)
                    self.vertices_select[index] = 1

                elif self.vertices_select[index] == 1:
                    self.vertices_move[index] = 1
                    self.moving_index = index
                    if not self.started:
                        self.moving_index_old = self.moving_index

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.moving_index >= 0:
                if self.vertices_move[self.moving_index] == 1:
                    a, b = x, y
                    self.img = self._img.copy()

                    # Indices of control points in array of selected points (vertices).
                    selected = np.where(self.vertices_select == 1)[0]

                    # Update the locations of the control points.
                    if not self.started:
                        locations = self.vertices[selected, :]
                    else:
                        locations = self.new_vertices[selected, :]

                    # TODO: make folowing hardcoded transformations to be dynamic
                    a = (a - settings.INT_DTYPE(640)) / settings.INT_DTYPE(-180)
                    b = (b - settings.INT_DTYPE(400)) / settings.INT_DTYPE(-180)

                    moving_c_index = np.where(selected == self.moving_index)[0][0]
                    # New location of control points after movement.
                    locations[moving_c_index, :] = np.array([a, b], dtype=settings.FLOAT_DTYPE)

                    if not np.array_equal(selected, self.selected_old):
                        self.started = True

                    # Transformation.
                    args = self.edges, self.vertices, self.gi, self.h, selected, locations
                    new_vertices, _, _ = StepOne.compute_v_prime(*args)
                    self.new_vertices = new_vertices[:]
                    self.draw_mesh(self.new_vertices, self.edges, self.img)

                    # Move control points in screen.
                    c_scaled = self.inscreen_scale(new_vertices[selected]).astype(int)
                    [cv2.circle(self.img, (vertex[0], vertex[1]), 5, (0, 0, 255), 1) for vertex in c_scaled]

        elif event == cv2.EVENT_LBUTTONUP:
            if self.vertices_move[self.moving_index] == 1:
                self.vertices_move[self.moving_index] = 0
                self.moving_index = -1

    def run(self):
        cv2.setUseOptimized(True)
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_event_callback, param=self)

        self.draw_mesh(self.vertices, self.edges, self.img)
        try:
            while True:
                cv2.imshow(self.window_name, self.img)
                # Press space bar to quit.
                if cv2.waitKey(1) == 32:
                    break
        except KeyboardInterrupt:
            sys.exit(0)
        finally:
            cv2.destroyAllWindows()


if __name__ == '__main__':
    puppet_path = '../data/puppet.obj'
    ui = Demo(puppet_path)
    ui.run()
