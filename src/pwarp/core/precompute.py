from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pwarp.core import ops
from pwarp.core.arap import StepOne

if TYPE_CHECKING:
    import numpy as np

__all__ = (
    "ArapPrecompute",
    "arap_precompute",
)


@dataclass(frozen=True)
class ArapPrecompute:
    """Mesh-level precomputed data for ARAP graph warp.

    This object contains everything that depends only on the mesh topology
    (faces, edges) and the original vertex positions used to build ARAP terms.

    It is intended to be created once and reused across many calls of graph_warp,
    especially in interactive demos (mouse dragging).
    """

    edges: np.ndarray
    gi: np.ndarray
    g_product: np.ndarray
    h: np.ndarray


def arap_precompute(
        vertices: np.ndarray,
        faces: np.ndarray,
        *,
        edges: np.ndarray | None = None,
) -> ArapPrecompute:
    """Build mesh-level ARAP precomputations.

    :param vertices: np.ndarray; original mesh vertices (N, 2)
    :param faces: np.ndarray; triangle indices (M, 3)
    :param edges: Optional[np.ndarray]; if provided, will be reused instead of recomputed
    :return: ArapPrecompute
    """
    if edges is None:
        edges = ops.get_edges(len(faces), faces)

    gi, g_product = StepOne.compute_g_matrix(vertices, edges, faces)
    h = StepOne.compute_h_matrix(edges, g_product, gi, vertices)

    return ArapPrecompute(
        edges=edges,
        gi=gi,
        g_product=g_product,
        h=h,
    )
