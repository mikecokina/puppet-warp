from typing import Tuple

from pwarp.core import ops
from pwarp.core import dtype
from pwarp import np

__all__ = (
    'StepOne',
    'StepTwo',
)


class StepOne(object):
    @staticmethod
    def compute_g_matrix(
            vertices: np.ndarray,
            edges: np.ndarray,
            faces: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
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
        g_product = np.zeros((np.size(edges, 0), 2, 8), dtype=dtype.FLOAT)
        gi = np.zeros((np.size(edges, 0), 4), dtype=dtype.FLOAT)

        if edges.dtype not in [dtype.INDEX]:
            raise ValueError('Invalid dtype of edge indices. Requires np.uint32, np.uint64 or int.')

        # Compute G_k matrix for each `k`.
        for k, edge in enumerate(edges):
            i_vert, j_vert = vertices[edge]
            i_index, j_index = edge

            l_index, r_index = ops.find_ijlr_vertices(edge, faces)
            l_vert = vertices[l_index]

            # For 3 neighbour points (when at the graph edge).
            if np.isnan(r_index):
                g = np.array([[i_vert[0], i_vert[1], 1, 0],
                              [i_vert[1], -i_vert[0], 0, 1],
                              [j_vert[0], j_vert[1], 1, 0],
                              [j_vert[1], -j_vert[0], 0, 1],
                              [l_vert[0], l_vert[1], 1, 0],
                              [l_vert[1], -l_vert[0], 0, 1]],
                             dtype=dtype.FLOAT)
                _slice = 6
            # For 4 neighbour points (when at the graph edge).
            else:
                r_vert = vertices[r_index]
                g = np.array([[i_vert[0], i_vert[1], 1, 0],
                              [i_vert[1], -i_vert[0], 0, 1],
                              [j_vert[0], j_vert[1], 1, 0],
                              [j_vert[1], -j_vert[0], 0, 1],
                              [l_vert[0], l_vert[1], 1, 0],
                              [l_vert[1], -l_vert[0], 0, 1],
                              [r_vert[0], r_vert[1], 1, 0],
                              [r_vert[1], -r_vert[0], 0, 1]],
                             dtype=dtype.FLOAT)
                _slice = 8

            # G[k,:,:]
            gi[k, :] = [i_index, j_index, l_index, np.nan if np.isnan(r_index) else r_index]
            x_matrix_pad = np.linalg.lstsq(g.T @ g, g.T, rcond=None)[0]
            g_product[k, :, :_slice] = x_matrix_pad[0:2]

        return gi, g_product

    @staticmethod
    def compute_h_matrix(
            edges: np.ndarray,
            g_product: np.ndarray,
            gi: np.ndarray,
            vertices: np.ndarray
    ) -> np.ndarray:
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
        h_matrix = np.zeros((np.size(edges, 0) * 2, 8), dtype=dtype.FLOAT)
        for k, edge in enumerate(edges):
            # ...where e is an edge vector..
            ek = np.subtract(*vertices[edge[::-1]])
            ek_matrix = np.array([[ek[0], ek[1]], [ek[1], -ek[0]]], dtype=dtype.FLOAT)

            # Ful llength of ones/zero matrix (will be sliced in case on the contour of graph).
            _oz = np.array([[-1, 0, 1, 0, 0, 0, 0, 0],
                            [0, -1, 0, 1, 0, 0, 0, 0]],
                           dtype=dtype.FLOAT)
            if np.isnan(gi[k, 3]):
                _slice = 6
            else:
                _slice = 8

            g = g_product[k, :, :_slice]
            oz = _oz[:, :_slice]
            h_calc = oz - (ek_matrix @ g)
            h_matrix[k * 2, :_slice] = h_calc[0]
            h_matrix[k * 2 + 1, :_slice] = h_calc[1]

        return h_matrix

    @staticmethod
    def compute_v_prime(
            edges: np.ndarray,
            vertices: np.ndarray,
            gi: np.ndarray,
            h_matrix: np.ndarray,
            c_indices: np.ndarray,
            c_vertices: np.ndarray,
            weight: dtype.FLOAT = dtype.FLOAT(1000.)
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        TODO:
            - make this method cacheable on A and b matrices.

        The cookbook from paper requires to compute expression `A1 @ v′ = b1`.
        Regards to the paper, we will compute v'.

        For more information see page 24 of paper on::

            https://www-ui.is.s.u-tokyo.ac.jp/~takeo/papers/takeo_jgt09_arapFlattening.pdf

        Warning: The position of h_{kuv} in matrix A of paper is based on position of given points (in k-th edge
        surounding points) in original vertices. It just demonstrates that 01, 23, 45, 67 indices of H row form a pair,
        but you have to put them on valid position or be aware of ordering of results in v' vector.

        :return: Tuple[np.ndarray, np.ndarray, np.ndarray]; (v_prime, A matrix, b vector)
        """
        # Prepare defaults.
        a1_matrix = np.zeros((np.size(edges, axis=0) * 2 + np.size(c_indices) * 2, np.size(vertices, axis=0) * 2),
                             dtype=dtype.FLOAT)
        b1_vector = np.zeros((np.size(edges, axis=0) * 2 + np.size(c_indices) * 2, 1), dtype=dtype.FLOAT)
        v_prime = np.zeros((np.size(vertices, axis=0), 2), dtype=dtype.FLOAT)

        # Fill values in prepared matrices/vectors
        for k, g_indices in enumerate(gi):
            for i, point_index in enumerate(g_indices):
                if not np.isnan(point_index):
                    point_index = int(point_index)
                    # In the h_matrix we have stored values for index k (edge index) in following form:
                    # for k = 0, two lines of h_matrix are going one after another like
                    # [k00, k10, k20, ..., k70] forlowed by [k01, k11, k21, ..., k71], hence we have to access values
                    # via (k * 2), and (k * 2 + 1).

                    # Variable point_index represent index from original vertices set of vertex from
                    # 4 neighbours of k-th edge.
                    # Index i represents an index of point from 4 (3 in case of contour) neighbours in k-th set.

                    # The row in the A matrix is defiend by index k. Since we have stored for given k index two rows
                    # of H matrix, we have to work with indexing of (k * 2) and (k * 2 + 1).

                    # The column of H matrix is accessible via index i, since H row is 0 - 7 indices long. Than
                    # (i * 2) and (i * 2 + 1) will access h valus for given point in given h row.

                    a1_matrix[k * 2, point_index * 2] = h_matrix[k * 2, i * 2]
                    a1_matrix[k * 2 + 1, point_index * 2] = h_matrix[k * 2 + 1, i * 2]
                    a1_matrix[k * 2, point_index * 2 + 1] = h_matrix[k * 2, i * 2 + 1]
                    a1_matrix[k * 2 + 1, point_index * 2 + 1] = h_matrix[k * 2 + 1, i * 2 + 1]

        for c_enum_index, c_vertex_index in enumerate(c_indices):
            # Set weights for given position of control point.
            a1_matrix[np.size(edges, axis=0) * 2 + c_enum_index * 2, c_vertex_index * 2] = weight
            a1_matrix[np.size(edges, axis=0) * 2 + c_enum_index * 2 + 1, c_vertex_index * 2 + 1] = weight
            # Do the same for values of b_vector
            b1_vector[np.size(edges, axis=0) * 2 + c_enum_index * 2] = weight * c_vertices[c_enum_index, 0]
            b1_vector[np.size(edges, axis=0) * 2 + c_enum_index * 2 + 1] = weight * c_vertices[c_enum_index, 1]

        v = np.linalg.lstsq(a1_matrix.T @ a1_matrix, a1_matrix.T @ b1_vector, rcond=None)[0]
        v_prime[:, 0] = v[0::2, 0]
        v_prime[:, 1] = v[1::2, 0]

        return v_prime, a1_matrix, b1_vector


class StepTwo(object):
    @staticmethod
    def compute_t_matrix(
            edges: np.ndarray,
            g_product: np.ndarray,
            gi: np.ndarray,
            v_prime: np.ndarray
    ) -> np.ndarray:
        """
        From paper:

        The second step takes the rotation information from the result of the first step
        (i.e., computing the explicit values of T′k and normalizing them to remove the
        scaling factor), rotates the original edge vectors ek by the amount T′k, and
        then solves Equation (1) using the original rotated edge vectors. That is, we
        compute the rotation of each edge by using the result of the first step,
        and then normalize it.
        
        :param edges: np.ndarray; 
        :param g_product: np.ndarray; 
        :param gi: np.ndarray;
        :param v_prime: np.ndarray; transformed point in sense of rotation from step one
        :return: np.ndarray;
        """
        t_matrix = np.zeros(((np.size(edges, axis=0)), 2, 2,), dtype=dtype.FLOAT)

        # We compute T′k for each edge.
        for k, edge in enumerate(edges):
            if np.isnan(gi[k, 3]):
                _slice = 6
                v = np.array([
                    [v_prime[int(gi[k, 0]), 0]],
                    [v_prime[int(gi[k, 0]), 1]],
                    [v_prime[int(gi[k, 1]), 0]],
                    [v_prime[int(gi[k, 1]), 1]],
                    [v_prime[int(gi[k, 2]), 0]],
                    [v_prime[int(gi[k, 2]), 1]]
                ], dtype=dtype.FLOAT)
            else:
                _slice = 8
                v = np.array([
                    [v_prime[int(gi[k, 0]), 0]],
                    [v_prime[int(gi[k, 0]), 1]],
                    [v_prime[int(gi[k, 1]), 0]],
                    [v_prime[int(gi[k, 1]), 1]],
                    [v_prime[int(gi[k, 2]), 0]],
                    [v_prime[int(gi[k, 2]), 1]],
                    [v_prime[int(gi[k, 3]), 0]],
                    [v_prime[int(gi[k, 3]), 1]]],
                    dtype=dtype.FLOAT
                )
            # We compute the rotation of each edge by using the result of the first step,
            g = g_product[k, :, :_slice]
            t = g @ v
            rot = np.array([[t[0], t[1]], [-t[1], t[0]]], dtype=dtype.FLOAT)
            # and then normalize it.
            t_normalized = (dtype.FLOAT(1) / np.sqrt(np.power(t[0], 2) + np.power(t[1], 2))) * rot
            # Store result.
            t_matrix[k, :, :] = t_normalized[:, :, 0]
        return t_matrix

    @staticmethod
    def compute_v_2prime(
            edges: np.ndarray,
            vertices: np.ndarray,
            t_matrix: np.ndarray,
            c_indices: np.ndarray,
            c_vertices: np.ndarray,
            weight: dtype.FLOAT = dtype.FLOAT(1000)
    ) -> np.ndarray:
        """

        :param edges: np.ndarray;
        :param vertices: np.ndarray;
        :param t_matrix: np.ndarray;
        :param c_indices: np.ndarray;
        :param c_vertices: np.ndarray;
        :param weight: np.float; dtype.FLOAT
        :return: np.ndarray;
        """
        # Prepare blueprints.
        a2_matrix = np.zeros((np.size(edges, axis=0) + np.size(c_indices), np.size(vertices, axis=0)),
                             dtype=dtype.FLOAT)
        b2_vector = np.zeros((np.size(edges, axis=0) + np.size(c_indices), 2), dtype=dtype.FLOAT)

        # Update values from precomputed components.
        # Matrix A2 is identical for both x- and y- components.
        for k, edge in enumerate(edges):
            # The values are set due to optimization equation from paper, where
            # arg min {sum_{i,j}||(v_j'' - v_i'')  ... ||} what gives 1 to position
            # of edge 0 and -1 to position of edge 1 and finally we will obtain (v_j'' - v_i'').
            a2_matrix[k, int(edge[0])] = dtype.FLOAT(-1.)
            a2_matrix[k, int(edge[1])] = dtype.FLOAT(1.)

            e = np.subtract(*vertices[edge[::-1]])
            t_e = t_matrix[k, :, :] @ e
            b2_vector[k, :] = t_e[0], t_e[1]

        for c_index, c in enumerate(c_indices):
            a2_matrix[np.size(edges, 0) + c_index, c] = weight
            b2_vector[np.size(edges, 0) + c_index, :] = weight * c_vertices[c_index, :]

        return np.linalg.lstsq(a2_matrix.T @ a2_matrix, a2_matrix.T @ b2_vector, rcond=None)[0]
