import unittest
import numpy as np

from pwarp.warp.affine import pad, unpad, affine_transformation


class TestAffine(unittest.TestCase):
    def test_pad_unpad_roundtrip(self):
        pts = np.array([[1.0, 2.0], [3.5, -4.25]], dtype=np.float64)

        h = pad(pts)
        self.assertEqual(h.shape, (2, 3))
        self.assertTrue(np.allclose(h[:, 2], 1.0))

        pts2 = unpad(h)
        self.assertEqual(pts2.shape, (2, 2))
        self.assertTrue(np.allclose(pts2, pts))

    def test_affine_transformation_maps_points_exact_for_affine_case(self):
        matrix_a = np.array(
            [
                [1.2, 0.3, 5.0],
                [-0.1, 0.9, -2.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        src = np.array(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 2.0], [2.0, 3.0]],
            dtype=np.float64,
        )

        src_h = np.hstack([src, np.ones((src.shape[0], 1), dtype=src.dtype)])
        dst_h = (matrix_a @ src_h.T).T
        dst = dst_h[:, :2]

        affine_t = affine_transformation(src, dst)
        self.assertEqual(affine_t.shape, (3, 3))

        pred = (affine_t @ src_h.T).T[:, :2]
        self.assertTrue(np.allclose(pred, dst, atol=1e-10))

    def test_affine_transformation_is_least_squares_when_overdetermined(self):
        rng = np.random.default_rng(0)

        src = rng.normal(size=(20, 2))

        matrix_a = np.array(
            [
                [0.8, -0.2, 1.0],
                [0.1, 1.1, 0.5],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        src_h = np.hstack([src, np.ones((src.shape[0], 1))])
        dst = (matrix_a @ src_h.T).T[:, :2] + 1e-6 * rng.normal(size=(20, 2))

        affine_t = affine_transformation(src, dst)
        pred = (affine_t @ src_h.T).T[:, :2]

        mean_err = float(np.mean(np.linalg.norm(pred - dst, axis=1)))
        self.assertLess(mean_err, 1e-4)

    def test_affine_warp_identity_smoke(self):
        # scikit-image is used in your affine.warp implementation
        # noinspection PyBroadException
        try:
            import skimage  # noqa: F401
        except Exception:
            self.skipTest("scikit-image not available")

        from pwarp.warp.affine import warp

        img = np.zeros((10, 12, 3), dtype=np.uint8)
        img[2:5, 3:7, :] = 255

        itentity = np.eye(3, dtype=np.float64)
        out = warp(img, itentity, output_shape=(10, 12), mode="constant", cval=0.0)

        self.assertEqual(out.shape, img.shape)
        self.assertTrue(np.allclose(out, img))


if __name__ == "__main__":
    unittest.main()
