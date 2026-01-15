import unittest
import numpy as np

from pwarp.core.precompute import arap_precompute
from pwarp.warp.warp import graph_warp
from pwarp.core import dtype


def _square_two_triangles_mesh():
    vertices = np.array(
        [
            [0.0, 1.0],  # 0
            [1.0, 1.0],  # 1
            [0.0, 0.0],  # 2
            [1.0, 0.0],  # 3
        ],
        dtype=dtype.FLOAT,
    )
    faces = np.array([[0, 2, 1], [1, 2, 3]], dtype=dtype.INT32)
    return vertices, faces


class TestGraphWarp(unittest.TestCase):
    def test_identity_when_controls_unchanged(self):
        v, f = _square_two_triangles_mesh()
        ctrl = np.array([0, 3], dtype=int)
        targets = v[ctrl].copy()

        out = graph_warp(v, f, ctrl, targets)
        self.assertEqual(out.shape, v.shape)
        self.assertTrue(np.isfinite(out).all())
        self.assertTrue(np.allclose(out[ctrl], targets, atol=1e-5))

    def test_moves_control_vertices_close_to_targets(self):
        v, f = _square_two_triangles_mesh()
        ctrl = np.array([0, 3], dtype=int)

        targets = v[ctrl].copy()
        targets[0] += np.array([-0.2, 0.1], dtype=dtype.FLOAT)
        targets[1] += np.array([0.15, -0.25], dtype=dtype.FLOAT)

        out = graph_warp(v, f, ctrl, targets)
        self.assertTrue(np.isfinite(out).all())
        self.assertTrue(np.allclose(out[ctrl], targets, atol=1e-3))

    def test_precompute_matches_non_precompute(self):
        v, f = _square_two_triangles_mesh()
        ctrl = np.array([0, 3], dtype=int)

        targets = v[ctrl].copy()
        targets[0] += np.array([-0.2, 0.1], dtype=dtype.FLOAT)

        out1 = graph_warp(v, f, ctrl, targets)

        pre = arap_precompute(v, f)
        out2 = graph_warp(v, f, ctrl, targets, precomputed=pre)

        self.assertTrue(np.allclose(out1, out2, atol=1e-8))

    def test_output_is_reasonable_bounded(self):
        v, f = _square_two_triangles_mesh()
        ctrl = np.array([0], dtype=int)
        targets = np.array([[5.0, 5.0]], dtype=dtype.FLOAT)

        out = graph_warp(v, f, ctrl, targets)
        self.assertTrue(np.isfinite(out).all())

        self.assertLess(float(np.max(np.abs(out))), 1e6)


if __name__ == "__main__":
    unittest.main()
