from typing import Union

from pwarp.core import ops
from pwarp.settings import settings
from pwarp import np


class StepOne(object):
    @staticmethod
    def compute_g_matrix(vertices: np.ndarray, edges: np.ndarray, faces: np.ndarray):
        """
        The paper requires to compute expression (G.T)^{-1} @ G.T = X.
        The problem might be solved by solving equation (G.T @ G) @ X = G.T, hence
        we can simply use np.linalg.lstsq(G.T @ G, G.T, ...).

        :param vertices: np.ndarray;
        :param edges: np.ndarray;
        :param faces: np.ndarray;
        :return: Tuple[np.ndarray, np.ndarray];

        ::

            gi represents indices of edges that contains result
            for expression (G.T)^{-1} @ G.T in g_product
        """
        g_product = np.zeros((np.size(edges, 0), 2, 8), dtype=settings.FLOAT_DTYPE)
        gi = np.zeros((np.size(edges, 0), 4), dtype=settings.FLOAT_DTYPE)

        if edges.dtype not in [settings.INDEX_DTYPE]:
            raise ValueError('Invalid dtype of edge indices. Requires np.uint32, np.uint64 or int.')

        # Compute G_k matrix for each `k`.
        for k, edge in enumerate(edges):
            i_vert, j_vert = vertices[edge].copy()
            i_index, j_index = edge

            l_index, r_index = ops.find_ijlr_vertices(edge, faces)
            l_vert = vertices[l_index].copy()

            # For 3 neighbour points (when at the graph edge).
            if np.isnan(r_index):
                g = np.array([[i_vert[0], i_vert[1], 1, 0],
                              [i_vert[1], -i_vert[0], 0, 1],
                              [j_vert[0], j_vert[1], 1, 0],
                              [j_vert[1], -j_vert[0], 0, 1],
                              [l_vert[0], l_vert[1], 1, 0],
                              [l_vert[1], -l_vert[0], 0, 1]],
                             dtype=settings.FLOAT_DTYPE)
                _slice = 6
            # For 4 neighbour points (when at the graph edge).
            else:
                r_vert = vertices[r_index].copy()
                g = np.array([[i_vert[0], i_vert[1], 1, 0],
                              [i_vert[1], -i_vert[0], 0, 1],
                              [j_vert[0], j_vert[1], 1, 0],
                              [j_vert[1], -j_vert[0], 0, 1],
                              [l_vert[0], l_vert[1], 1, 0],
                              [l_vert[1], -l_vert[0], 0, 1],
                              [r_vert[0], r_vert[1], 1, 0],
                              [r_vert[1], -r_vert[0], 0, 1]],
                             dtype=settings.FLOAT_DTYPE)
                _slice = 8

            # G[k,:,:]
            gi[k, :] = [i_index, j_index, l_index, r_index]
            x_matrix_pad = np.linalg.lstsq(g.T @ g, g.T, rcond=None)[0]
            g_product[k, :, :_slice] = x_matrix_pad[0:2, :]

        return gi, g_product

    @staticmethod
    def compute_h_matrix(edges: np.ndarray, g_product: np.ndarray, gi: np.ndarray, vertices: np.ndarray):
        """
        Transformed term (v′_j − v′_i) − T_{ij} (v_j − v_i) from paper requires
        computation of matrix H. To be able compute matrix H, we need matrix G
        from other method.

        :param edges: np.ndarray; requires dtype int/np.uint32/np.uint64
        :param g_product: np.ndarray;
        :param gi: np.ndarray;
        :param vertices: np.ndarray;
        :return: np.ndarray;
        """
        h_matrix = np.zeros((np.size(edges, 0) * 2, 8), dtype=settings.FLOAT_DTYPE)
        for k, edge in enumerate(edges):
            # ...where e is an edge vector..
            ek = np.subtract(*vertices[edge[::-1]])
            ek_matrix = np.array([[ek[0], ek[1]], [ek[1], -ek[0]]], dtype=settings.FLOAT_DTYPE)

            # Ful llength of ones/zero matrix (will be sliced in case on the contour of graph).
            _oz = np.array([[-1, 0, 1, 0, 0, 0, 0, 0],
                            [0, -1, 0, 1, 0, 0, 0, 0]],
                           dtype=settings.FLOAT_DTYPE)
            if np.isnan(gi[k, 3]):
                _slice = 6
            else:
                _slice = 8

            g = g_product[k, :, :_slice]
            oz = _oz[:, :_slice]
            h_calc = oz - (ek_matrix @ g)
            h_matrix[k * 2, :_slice] = h_calc[0, :]
            h_matrix[k * 2 + 1, :_slice] = h_calc[1, :]

        return h_matrix


class StepTwo(object):
    pass


if __name__ == '__main__':
    from pwarp._io import read_wavefront
    from matplotlib import pyplot as plt

    _nr, _nf, _r, _f = read_wavefront('../data/puppet.obj')
    _edges = ops.get_edges(_nf, _f)
    _edge = _edges[161]
    _gi, _g_product = StepOne.compute_g_matrix(_r, _edges, _f)
    StepOne.compute_h_matrix(_edges, _g_product, _gi, _r)

    # _l_, _r_ = find_ijlr_vertices(_edge, _f, _r)
    #
    # fig, ax = plt.subplots()
    # plt.tight_layout(pad=0)
    # ax.triplot(_r.T[0], _r.T[1], _f)
    # ax.set_aspect('equal')
    # ax.axis('off')
    #
    # ax.scatter(_r[_edge].T[0], _r[_edge].T[1], c="r", s=20)
    # ax.scatter(_l_.T[0], _l_.T[1], c="g", s=20)
    # if not np.all(np.isnan(_r_)):
    #     ax.scatter(_r_.T[0], _r_.T[1], c="k", s=20)
    # plt.show()
