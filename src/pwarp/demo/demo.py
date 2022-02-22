import os
import sys
import os.path as op

import cv2

from pwarp import np, _io
from pwarp.core import ops, dtype
from pwarp.core.arap import StepOne, StepTwo
from pwarp.demo import misc
from pwarp.ui import draw
from pwarp.logger import getLogger

logger = getLogger("pwarp.demo.demo")


class Demo(object):
    """
    Puppet Warp Demo
    ================

    FAQ:

        - Q: How to add control points?
        - A: Click LMB on any top of the triangle.

        - Q: How to exit application.
        - A: Press Esc to quit.

        - Q: How to save transformed mesh?
        - A: Press Space Bar to save transformed mesh in wavefron format.

        - Q: Where is output stored?
        - A: The each saved output is by default stored in directory ~/pwarp.

        - Q: Is it possible to configure output directory?
        - A: Yes, you can change an output directory via initialization variable `output_dir`.

    """
    def __init__(
            self,
            obj_path: str,
            window_name: str = 'ARAP',
            screen_width: int = 1280,
            screen_height: int = 800,
            scale: int = -180,
            output_dir: str = None
    ):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.dx = dtype.INT(self.screen_width // 2)
        self.dy = dtype.INT(self.screen_height // 2)
        self.scale = dtype.INT(scale)

        if output_dir is None:
            output_dir = op.expanduser("~/pwarp")
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

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
        self.new_vertices = np.array([])

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

        self.img = np.zeros((self.screen_height, self.screen_width, 3), np.uint8)
        self._img = self.img.copy()

        # Compute initial transformation matrices
        self.gi, self.g_product = StepOne.compute_g_matrix(self.vertices, self.edges, self.faces)
        self.h = StepOne.compute_h_matrix(self.edges, self.g_product, self.gi, self.vertices)

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

                    a = (a - dtype.INT(self.dx)) / dtype.INT(self.scale)
                    b = (b - dtype.INT(self.dy)) / dtype.INT(self.scale)

                    # Moving control index is index appliable to array `selected`.
                    moving_c_index = np.where(selected == self.moving_index)[0][0]
                    # New location of control points after movement.
                    locations[moving_c_index, :] = np.array([a, b], dtype=dtype.FLOAT)

                    if not np.array_equal(selected, self.selected_old):
                        self.started = True

                    # Transformation.
                    args = self.edges, self.vertices, self.gi, self.h, selected, locations
                    new_vertices, _, _ = StepOne.compute_v_prime(*args)
                    t_matrix = StepTwo.compute_t_matrix(self.edges, self.g_product, self.gi, new_vertices)
                    new_vertices = StepTwo.compute_v_2prime(self.edges, self.vertices, t_matrix, selected, locations)
                    self.new_vertices = new_vertices[:]
                    draw.draw_mesh(self.new_vertices, self.edges, self.img, self.dx, self.dy, self.scale)

                    # Move control points in screen.
                    c_scaled = draw.shift_scale(new_vertices[selected], self.dx, self.dy, self.scale).astype(int)
                    [cv2.circle(self.img, (vertex[0], vertex[1]), 5, (0, 0, 255), 1) for vertex in c_scaled]

        elif event == cv2.EVENT_LBUTTONUP:
            if self.vertices_move[self.moving_index] == 1:
                self.vertices_move[self.moving_index] = 0
                self.moving_index = -1

    def run(self):
        cv2.setUseOptimized(True)
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_event_callback, param=self)

        draw.draw_mesh(self.vertices, self.edges, self.img, self.dx, self.dy, self.scale)
        try:
            count = 0
            while True:
                cv2.imshow(self.window_name, self.img)
                # Press esc bar to quit.
                key = cv2.waitKey(1)
                if key == 27:
                    break
                # Press space bar to save.
                elif key == 32:
                    if np.size(self.new_vertices, axis=0) == 0:
                        continue

                    path = op.join(self.output_dir, f'puppet_{count:03d}.obj')
                    _io.save_wavefront(path, self.num_vertices, self.num_faces, self.new_vertices, self.faces)
                    logger.info(f'object saved as {path}')
                    count += 1

        except KeyboardInterrupt:
            logger.info("quit")
            sys.exit(0)
        finally:
            cv2.destroyAllWindows()


if __name__ == '__main__':
    puppet_path = '../data/puppet.obj'
    ui = Demo(puppet_path)
    ui.run()