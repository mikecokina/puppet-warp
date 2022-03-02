from pwarp import np
from pwarp.core import dtype


def find_ijlr_vertices(edge: np.ndarray, faces: np.ndarray):
    """
    The rotation matrix Tk is given as a transformation that maps the vertices
    around the edge to new positions as closely as possible in a least-squares
    sense. WE SAMPLE FOUR VERTICES around the edge as a context to derive the
    local transformation Tk. It is possible to sample an arbitrary number of
    vertices greater than three here, but four is the most straightforward, and we
    have found that it produces good results. An exception applies to edges on
    the boundary. In those cases, we only USE THREE VERTICES to compute Tk.

    Source: Igarashi et al., 2009

    ::

        l x--------x j
                // edge
            i x-------x r

    Find indices of vertex l and r. When edge i-j is situated at the edge of graph,
    than use only vertex l and r will be set as np.nan.

    :param edge: np.ndarray;
    :param faces: np.ndarray;
    :return: np.ndarray;
    """

    lr_indices = [np.nan, np.nan]
    count = 0
    for i, face in enumerate(faces):
        if np.any(face == edge[0]):
            if np.any(face == edge[1]):
                neighbour_index = np.where(face[np.where(face != edge[0])] != edge[1])[0][0]
                n = face[np.where(face != edge[0])]
                lr_indices[count] = int(n[neighbour_index])
                count += 1

                if count == 2:
                    break
    l_index, r_index = lr_indices
    return [l_index, r_index]


def get_edges(no_faces: int, faces: np.ndarray):
    """
    Find all edges from given faces.

    :param no_faces: int;
    :param faces: np.ndarray;
    :return: np.ndarray;
    """
    edges = np.zeros([no_faces * 3, 2])
    for index, face in enumerate(faces):
        edges[index * 3, :] = [face[0], face[1]]
        edges[index * 3 + 1, :] = [face[1], face[2]]
        edges[index * 3 + 2, :] = [face[0], face[2]]
    edges.sort(axis=1)
    edges = np.unique(edges, axis=0)
    return np.array(edges, dtype=dtype.INDEX)
