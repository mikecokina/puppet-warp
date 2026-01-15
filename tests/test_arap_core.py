import unittest
import numpy as np

from pwarp.core.arap import StepOne, StepTwo
from pwarp.core.const import NP_INAN
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

    edges = np.array(
        [
            [0, 1],
            [1, 3],
            [3, 2],
            [2, 0],
            [1, 2],
        ],
        dtype=dtype.INDEX,
    )

    return vertices, faces, edges


class TestArapCore(unittest.TestCase):
    def test_compute_g_matrix_rejects_non_integer_edges(self):
        v, f, e = _square_two_triangles_mesh()
        e_bad = e.astype(np.float64)
        with self.assertRaises(ValueError):
            StepOne.compute_g_matrix(v, e_bad, f)

    def test_compute_g_matrix_shapes_and_gi_contents(self):
        v, f, e = _square_two_triangles_mesh()
        gi, gprod = StepOne.compute_g_matrix(v, e, f)

        self.assertEqual(gi.shape, (e.shape[0], 4))
        self.assertEqual(gprod.shape, (e.shape[0], 2, 8))

        self.assertTrue(np.all(gi[:, 0] >= 0))
        self.assertTrue(np.all(gi[:, 1] >= 0))

        # some boundary edges should have missing r_index
        self.assertTrue(bool(np.any(gi[:, 3] == NP_INAN)))

    def test_compute_h_matrix_shape_and_boundary_slice_behavior(self):
        v, f, e = _square_two_triangles_mesh()
        gi, gprod = StepOne.compute_g_matrix(v, e, f)
        h = StepOne.compute_h_matrix(e, gprod, gi, v)

        self.assertEqual(h.shape, (e.shape[0] * 2, 8))

        boundary = np.where(gi[:, 3] == NP_INAN)[0]
        if boundary.size:
            k = int(boundary[0])
            self.assertTrue(np.allclose(h[k * 2: k * 2 + 2, 6:8], 0.0))

    def test_build_a1_b1_dimensions_and_constraint_rows(self):
        v, f, e = _square_two_triangles_mesh()
        gi, gprod = StepOne.compute_g_matrix(v, e, f)
        h = StepOne.compute_h_matrix(e, gprod, gi, v)

        c_idx = np.array([0, 3], dtype=int)
        c_v = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=dtype.FLOAT)
        w = dtype.FLOAT(1000.0)

        A1 = StepOne.build_a_matrix(e, v, gi, h, c_idx, weight=w)
        b1 = StepOne.build_b_vector(e, c_idx, c_v, weight=w)

        self.assertEqual(A1.shape, (e.shape[0] * 2 + c_idx.shape[0] * 2, v.shape[0] * 2))
        self.assertEqual(b1.shape, (e.shape[0] * 2 + c_idx.shape[0] * 2, 1))

        base = e.shape[0] * 2

        self.assertEqual(A1[base + 0, 0 * 2], w)
        self.assertEqual(A1[base + 1, 0 * 2 + 1], w)
        self.assertEqual(A1[base + 2, 3 * 2], w)
        self.assertEqual(A1[base + 3, 3 * 2 + 1], w)

        self.assertEqual(b1[base + 0, 0], w * c_v[0, 0])
        self.assertEqual(b1[base + 1, 0], w * c_v[0, 1])

    def test_step1_solve_step_returns_finite(self):
        v, f, e = _square_two_triangles_mesh()
        gi, gprod = StepOne.compute_g_matrix(v, e, f)
        h = StepOne.compute_h_matrix(e, gprod, gi, v)

        c_idx = np.array([0, 3], dtype=int)
        c_v = np.array([[0.1, 1.2], [1.1, -0.1]], dtype=dtype.FLOAT)

        v_prime, A1, b1 = StepOne.compute_v_prime(e, v, gi, h, c_idx, c_v)

        self.assertEqual(v_prime.shape, v.shape)
        self.assertTrue(np.isfinite(v_prime).all())
        self.assertEqual(A1.shape[0], b1.shape[0])

    def test_build_a2_b2_dimensions(self):
        v, f, e = _square_two_triangles_mesh()
        c_idx = np.array([0, 3], dtype=int)
        w = dtype.FLOAT(1000.0)

        A2 = StepTwo.build_a2_matrix(e, v, c_idx, weight=w)
        self.assertEqual(A2.shape, (e.shape[0] + c_idx.shape[0], v.shape[0]))

        for k, (i, j) in enumerate(e):
            self.assertEqual(A2[k, int(i)], dtype.FLOAT(-1.0))
            self.assertEqual(A2[k, int(j)], dtype.FLOAT(1.0))

        base = e.shape[0]
        self.assertEqual(A2[base + 0, int(c_idx[0])], w)
        self.assertEqual(A2[base + 1, int(c_idx[1])], w)

    def test_compute_t_matrix_is_rotation_like(self):
        v, f, e = _square_two_triangles_mesh()
        gi, gprod = StepOne.compute_g_matrix(v, e, f)
        h = StepOne.compute_h_matrix(e, gprod, gi, v)

        c_idx = np.array([0, 3], dtype=int)
        c_v = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=dtype.FLOAT)
        v_prime, _, _ = StepOne.compute_v_prime(e, v, gi, h, c_idx, c_v)

        T = StepTwo.compute_t_matrix(e, gprod, gi, v_prime)
        self.assertEqual(T.shape, (e.shape[0], 2, 2))

        for k in range(T.shape[0]):
            R = T[k]
            RtR = R.T @ R
            self.assertTrue(np.allclose(RtR, np.eye(2), atol=1e-4, rtol=1e-4))

            det = float(np.linalg.det(R))
            self.assertTrue(np.isfinite(det))
            self.assertGreater(det, 0.0)


if __name__ == "__main__":
    unittest.main()
