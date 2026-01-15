import unittest

import numpy as np

from pwarp.mesh.tri import triangular_mesh


class TestTriangularMesh(unittest.TestCase):
    def test_rejects_non_positive_delta(self):
        with self.assertRaises(ValueError):
            triangular_mesh(width=10, height=10, delta=0)
        with self.assertRaises(ValueError):
            triangular_mesh(width=10, height=10, delta=-1)

    def test_rejects_non_positive_dims(self):
        with self.assertRaises(ValueError):
            triangular_mesh(width=0, height=10, delta=5)
        with self.assertRaises(ValueError):
            triangular_mesh(width=10, height=0, delta=5)
        with self.assertRaises(ValueError):
            triangular_mesh(width=-10, height=10, delta=5)

    def test_rejects_unknown_method(self):
        with self.assertRaises(ValueError):
            triangular_mesh(width=30, height=20, delta=10, method="nope")

    def test_scipy_basic_contract(self):
        v, f = triangular_mesh(width=40, height=30, delta=10, method="scipy")

        self.assertIsInstance(v, np.ndarray)
        self.assertIsInstance(f, np.ndarray)

        self.assertEqual(v.ndim, 2)
        self.assertEqual(v.shape[1], 2)

        self.assertEqual(f.ndim, 2)
        self.assertEqual(f.shape[1], 3)

        self.assertGreaterEqual(int(f.min()), 0)
        self.assertLess(int(f.max()), v.shape[0])

        # bbox (tolerance for floating error)
        self.assertGreaterEqual(float(v[:, 0].min()), -1e-6)
        self.assertGreaterEqual(float(v[:, 1].min()), -1e-6)
        self.assertLessEqual(float(v[:, 0].max()), 40.0 + 1e-6)
        self.assertLessEqual(float(v[:, 1].max()), 30.0 + 1e-6)


if __name__ == "__main__":
    unittest.main()
