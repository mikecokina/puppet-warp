from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from pwarp.warp.warp import graph_warp
from pwarp.core.precompute import arap_precompute


class TestE2ESmallMesh(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        data_dir = Path(__file__).resolve().parent / "data"

        cls.vertices = np.load(data_dir / "small_vertices.npy")
        cls.faces = np.load(data_dir / "small_faces.npy")
        cls.control_indices = np.load(data_dir / "small_control_indices.npy")
        cls.shifted_locations = np.load(data_dir / "small_shifted_locations.npy")
        cls.expected = np.load(data_dir / "small_warped_expected.npy")

    def test_graph_warp_matches_saved_expected(self):
        pre = arap_precompute(vertices=self.vertices, faces=self.faces)

        warped = graph_warp(
            vertices=self.vertices,
            faces=self.faces,
            control_indices=self.control_indices,
            shifted_locations=self.shifted_locations,
            precomputed=pre,
        )

        self.assertEqual(warped.shape, self.expected.shape)
        self.assertTrue(np.isfinite(warped).all())

        # Controls should match their shifted targets closely
        self.assertTrue(
            np.allclose(
                warped[self.control_indices],
                self.shifted_locations,
                atol=1e-5,
                rtol=0.0,
            )
        )

        # Full regression check
        # If CI has different BLAS/SVD behavior, bump atol to 1e-6.
        self.assertTrue(
            np.allclose(
                warped,
                self.expected,
                atol=1e-7,
                rtol=1e-7,
            )
        )

    def test_graph_warp_without_precompute_still_matches(self):
        warped = graph_warp(
            vertices=self.vertices,
            faces=self.faces,
            control_indices=self.control_indices,
            shifted_locations=self.shifted_locations,
            precomputed=None,
        )

        self.assertTrue(np.isfinite(warped).all())

        self.assertTrue(
            np.allclose(
                warped,
                self.expected,
                atol=1e-7,
                rtol=1e-7,
            )
        )


if __name__ == "__main__":
    unittest.main()
