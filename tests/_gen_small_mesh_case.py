from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import matplotlib as mpl

from pwarp.warp.warp import graph_warp
from pwarp.core.precompute import arap_precompute


def _configure_matplotlib_backend() -> None:
    """Configure a Matplotlib backend for interactive use or fallback rendering."""
    backend = os.environ.get("MPL_BACKEND", "").strip()
    if backend:
        mpl.use(backend, force=True)
        return

    # Try stable interactive backends first; fallback to Agg.
    for candidate in ("TkAgg", "QtAgg", "Agg"):
        # noinspection PyBroadException
        try:
            mpl.use(candidate, force=True)
        except Exception:  # pragma: no cover
            continue
        else:
            return


_configure_matplotlib_backend()


def make_small_mesh() -> tuple[np.ndarray, np.ndarray]:
    """
    3x3 grid on unit square, split into triangles.

    6--7--8
    | /| /|
    3--4--5
    | /| /|
    0--1--2
    """
    xs = [0.0, 0.5, 1.0]
    ys = [0.0, 0.5, 1.0]

    vertices = np.array([[x, y] for y in ys for x in xs], dtype=float)

    def idx(ix: int, iy: int) -> int:
        return iy * 3 + ix

    faces = []
    for iy in range(2):
        for ix in range(2):
            v00 = idx(ix, iy)
            v10 = idx(ix + 1, iy)
            v01 = idx(ix, iy + 1)
            v11 = idx(ix + 1, iy + 1)

            faces.append([v00, v10, v11])
            faces.append([v00, v11, v01])

    return vertices, np.asarray(faces, dtype=np.int32)


def plot_mesh_pair(
    vertices: np.ndarray,
    faces: np.ndarray,
    warped: np.ndarray,
    ctrl_idx: np.ndarray,
    ctrl_dst: np.ndarray,
) -> None:
    tri_src = mtri.Triangulation(vertices[:, 0], vertices[:, 1], faces)
    tri_dst = mtri.Triangulation(warped[:, 0], warped[:, 1], faces)

    _fig, ax = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)

    ax[0].set_title("Original")
    ax[0].triplot(tri_src, linewidth=0.8)
    ax[0].scatter(vertices[ctrl_idx, 0], vertices[ctrl_idx, 1], s=50)

    ax[1].set_title("Warped")
    ax[1].triplot(tri_dst, linewidth=0.8)
    ax[1].scatter(ctrl_dst[:, 0], ctrl_dst[:, 1], s=50, color="red")

    # arrows
    src = vertices[ctrl_idx]
    ax[1].quiver(
        src[:, 0],
        src[:, 1],
        ctrl_dst[:, 0] - src[:, 0],
        ctrl_dst[:, 1] - src[:, 1],
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.01,
    )

    for a in ax:
        a.set_aspect("equal")
        a.grid(True, alpha=0.3)

    plt.show()


def main() -> None:
    vertices, faces = make_small_mesh()

    # Pick 4 corners as controls
    control_indices = np.array([0, 2, 6, 8], dtype=int)

    shifted_locations = vertices[control_indices].copy()
    shifted_locations[0] += [-0.15, -0.10]
    shifted_locations[1] += [0.10, -0.05]
    shifted_locations[2] += [-0.05, 0.12]
    shifted_locations[3] += [0.15, 0.08]

    # Optional but realistic: precompute once
    pre = arap_precompute(vertices=vertices, faces=faces)

    warped = graph_warp(
        vertices=vertices,
        faces=faces,
        control_indices=control_indices,
        shifted_locations=shifted_locations,
        precomputed=pre,
    )

    print("Warped vertices:")
    np.set_printoptions(precision=6, suppress=True)
    print(warped)

    # Save to tests/data
    data_dir = Path(__file__).resolve().parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    np.save(data_dir / "small_vertices.npy", vertices)
    np.save(data_dir / "small_faces.npy", faces)
    np.save(data_dir / "small_control_indices.npy", control_indices)
    np.save(data_dir / "small_shifted_locations.npy", shifted_locations)
    np.save(data_dir / "small_warped_expected.npy", warped)

    print(f"Saved .npy files to: {data_dir}")

    plot_mesh_pair(
        vertices=vertices,
        faces=faces,
        warped=warped,
        ctrl_idx=control_indices,
        ctrl_dst=shifted_locations,
    )


if __name__ == "__main__":
    main()
