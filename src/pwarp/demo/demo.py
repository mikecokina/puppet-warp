from __future__ import annotations

import sys
from pathlib import Path

import cv2

from pwarp import _io, np
from pwarp.core import dtype
from pwarp.core.precompute import arap_precompute
from pwarp.demo import draw, misc
from pwarp.logger import getLogger
from pwarp.warp import warp
from pwarp.warp.warp import graph_warp

logger = getLogger("pwarp.demo.demo")


class Demo:
    """Puppet Warp Demo.

    FAQ:

        - Q: How to add control points?
        - A: Click LMB on any top of the triangle.

        - Q: How to exit application.
        - A: Press Esc to quit.

        - Q: How to save transformed mesh?
        - A: Press Space Bar to save transformed mesh in wavefron format.

        - Q: Where is output stored?
        - A: Each saved output is by default stored in directory ~/pwarp.

        - Q: Is it possible to configure output directory?
        - A: Yes, you can change an output directory via initialization variable `output_dir`.
    """

    def __init__(
            self,
            obj_path: str | Path = Path(__file__).parent.parent / "data" / "puppet.obj",
            window_name: str = "ARAP",
            screen_width: int = 1280,
            screen_height: int = 800,
            scale: float = -200,
            dx: int | None = None,
            dy: int | None = None,
            output_dir: str | Path | None = None,
            image: str | Path | None = Path(__file__).parent.parent / "data" / "puppet.png",
            bg_fill: int | tuple[int, int, int] | list[int] = 255,
            *,
            verbose: bool = False,
    ) -> None:
        """Initialize the interactive Puppet Warp demo.

        Load the mesh and optional background image, initialize interaction state,
        prepare screen-space transforms, and precompute mesh-level ARAP data.

        :param obj_path: Path to the Wavefront OBJ mesh.
        :param window_name: Name of the OpenCV window.
        :param screen_width: Width of the rendering window in pixels.
        :param screen_height: Height of the rendering window in pixels.
        :param scale: Scaling factor applied to mesh coordinates.
        :param dx: Horizontal screen offset. Defaults to half screen width.
        :param dy: Vertical screen offset. Defaults to half screen height.
        :param output_dir: Directory used to store exported OBJ files.
        :param image: Optional background image path.
        :param bg_fill: Background fill color for image warping.
        :param verbose: Enable verbose logging output.
        """
        # Screen dimensions.
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.dx = dtype.INT(self.screen_width // 2) if dx is None else dx
        self.dy = dtype.INT(self.screen_height // 2) if dy is None else dy
        self.scale = dtype.FLOAT(scale)

        # Background image preparation.
        self.img = np.zeros((self.screen_height, self.screen_width, 3), np.uint8)
        self.transform_image = False

        if image is not None:
            image_path = Path(image)
            if not image_path.is_file():
                msg = f"No image file {image_path}"
                raise FileNotFoundError(msg)
            self.img = cv2.imread(str(image_path))
            self.transform_image = True

        self._img = self.img.copy()
        self._transformed_background = self.img.copy()

        # Misc.
        self._bg_fill = bg_fill

        if output_dir is None:
            output_dir = Path.home() / "pwarp"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Default interaction attributes.
        self.start_select = False
        self.end_select = False
        self.start_move = False
        self.end_move = False
        self.moving_index = -1
        self.selected_old = np.array([-1], dtype=int)
        self.a_matrix = None
        self.b_vector = None
        self.moving_index_old = -1
        self.started = False
        self.new_vertices = np.array([])
        self.window_name = window_name

        self._circle_radius = 5
        self._verbose = verbose

        # Load Wavefront object.
        obj_path = Path(obj_path)
        num_vertices, num_faces, vertices, faces = _io.read_wavefront(obj_path)
        self.num_vertices = num_vertices
        self.num_faces = num_faces
        self.vertices = vertices
        self.faces = faces

        self.vertices_move = np.zeros(self.num_vertices)
        self.vertices_select = np.zeros(self.num_vertices)

        # Original vertices transformed to image dimensions.
        self.vertices_t = self.shift_and_scale(self.vertices)

        # Mesh-level ARAP precompute belongs to core. Demo stores and reuses it.
        self.pre = arap_precompute(vertices=self.vertices, faces=self.faces)
        self.edges = self.pre.edges

    def shift_and_scale(self, vertices: np.ndarray) -> np.ndarray:
        """Shift and scale vertices to screen coordinates.

        :param vertices: Vertex array of shape (N, 2).
        :return: Integer vertex array transformed for screen drawing.
        """
        vertices = vertices.copy()
        vertices[:, 0] = vertices[:, 0] * self.scale + self.dx
        vertices[:, 1] = vertices[:, 1] * self.scale + self.dy
        return vertices.astype(int)

    @staticmethod
    def mouse_event_callback(  # noqa: C901, PLR0912, PLR0915
            event: int,
            x: int,
            y: int,
            _flags: int,
            param: Demo,
    ) -> None:
        """Handle OpenCV mouse events.

        :param event: OpenCV event code.
        :param x: Mouse x coordinate in window pixels.
        :param y: Mouse y coordinate in window pixels.
        :param _flags: OpenCV event flags.
        :param param: User parameter passed via cv2.setMouseCallback.
        """
        self: Demo = param  # type: ignore[assignment]

        if event == cv2.EVENT_LBUTTONDOWN:
            is_close, index = misc.is_close(x, y, self.shift_and_scale(self.vertices))

            if self.started:
                is_close, index = misc.is_close(x, y, self.shift_and_scale(self.new_vertices))

            if is_close:
                # Mark vertex as control point.
                if self.vertices_select[index] == 0:
                    # If already manipulated, new vertices requires rescaling, since data are stored in original scale.
                    if len(self.new_vertices):
                        selected_vertex = self.new_vertices[index]
                        selected_vertex = self.shift_and_scale(np.array([selected_vertex]))[0]
                    else:
                        selected_vertex = self.vertices_t[index].astype(int)

                    self.vertices_select[index] = 1

                    _x, _y = selected_vertex
                    cv2.circle(self.img, (_x, _y), self._circle_radius, (0, 0, 255), 1)
                    self.vertices_select[index] = 1

                elif self.vertices_select[index] == 1:
                    self.vertices_move[index] = 1
                    self.moving_index = index
                    if not self.started:
                        self.moving_index_old = self.moving_index

        elif event == cv2.EVENT_MBUTTONDOWN:
            # Deselect point with double click.
            is_close, index = misc.is_close(x, y, self.shift_and_scale(self.vertices))

            if self.started:
                is_close, index = misc.is_close(x, y, self.shift_and_scale(self.new_vertices))

            is_selected = self.vertices_select[index] == 1
            if is_close and is_selected:
                self.vertices_select[index] = 0

                # Redraw.
                vertices = self.new_vertices if len(self.new_vertices) else self.vertices_t
                _x, _y = self.shift_and_scale(vertices)[index]
                radius = self._circle_radius

                content = self._transformed_background[_y - radius:_y + radius, _x - radius:_x + radius]
                self.img[_y - radius:_y + radius, _x - radius:_x + radius] = content
                draw.draw_mesh(self.shift_and_scale(vertices), self.edges, self.img)
                cv2.imshow(self.window_name, self.img)

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.moving_index >= 0 and self.vertices_move[self.moving_index] == 1:
                a, b = x, y
                self.img = self._img.copy()

                # Indices of control points in array of selected points (vertices).
                selected = np.where(self.vertices_select == 1)[0]

                # Update the locations of the control points.
                locations = self.vertices[selected, :] if not self.started else self.new_vertices[selected, :]

                if self._verbose:
                    logger.info("Control points: %s", selected)
                    logger.info("Control points coordinates: %s", locations)

                # noinspection PyUnresolvedReferences
                a = (a - dtype.FLOAT(self.dx)) / dtype.FLOAT(self.scale)
                # noinspection PyUnresolvedReferences
                b = (b - dtype.FLOAT(self.dy)) / dtype.FLOAT(self.scale)

                # Moving control index is index appliable to array `selected`.
                moving_c_index = np.where(selected == self.moving_index)[0][0]
                # New location of control points after movement.
                locations[moving_c_index, :] = np.array([a, b], dtype=dtype.FLOAT)

                if self._verbose:
                    logger.info("Control points moved to position %s", locations)

                if not np.array_equal(selected, self.selected_old):
                    self.started = True
                    self.selected_old = selected.copy()

                # Transformation (orchestration is in graph_warp, math stays in core).
                new_vertices = graph_warp(
                    vertices=self.vertices,
                    faces=self.faces,
                    control_indices=selected,
                    shifted_locations=locations,
                    precomputed=self.pre,
                )
                self.new_vertices = new_vertices[:]

                new_vertices_t = self.shift_and_scale(new_vertices)
                if self.transform_image:
                    self.img = warp.graph_defined_warp(
                        self.img,
                        self.vertices_t,
                        self.faces,
                        new_vertices_t,
                        self.faces,
                        bg_fill=self._bg_fill,
                    )
                    self._transformed_background = self.img.copy()

                draw.draw_mesh(self.shift_and_scale(self.new_vertices), self.edges, self.img)

                # Move control points in screen.
                [
                    cv2.circle(self.img, (vertex[0], vertex[1]), self._circle_radius, (0, 0, 255), 1)
                    for vertex in new_vertices_t[selected]
                ]

        elif event == cv2.EVENT_LBUTTONUP:
            if self.vertices_move[self.moving_index] == 1:
                self.vertices_move[self.moving_index] = 0
                self.moving_index = -1

    def run(self) -> None:
        """Run the OpenCV demo loop."""
        cv2.setUseOptimized(True)
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_event_callback, param=self)

        draw.draw_mesh(self.vertices_t, self.edges, self.img)
        try:
            count = 0
            while True:
                cv2.imshow(self.window_name, self.img)
                # Press esc bar to quit.
                key = cv2.waitKey(1)
                if key == 27:
                    break
                # Press space bar to save.
                if key == 32:
                    if np.size(self.new_vertices, axis=0) == 0:
                        continue

                    path = Path(self.output_dir) / f"puppet_{count:03d}.obj"
                    _io.save_wavefront(path, self.num_vertices, self.num_faces, self.new_vertices, self.faces)
                    logger.info("object saved as %s", path)
                    count += 1

        except KeyboardInterrupt:
            logger.info("quit")
            sys.exit(0)
        finally:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    Demo().run()
