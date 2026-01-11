from pwarp import np
from pwarp.core import dtype


def build_edge_opposites(faces: np.ndarray) -> dict[tuple[int, int], list[int]]:
    """
    Precompute adjacency information for fast edge queries.

    Builds a mapping from undirected edge (min(i, j), max(i, j)) to a list of
    opposite vertices from adjacent faces.

    This is a performance optimization to avoid scanning all faces when
    resolving local neighborhoods around edges.
    """
    faces = np.asarray(faces, dtype=int)

    edge_to_opp: dict[tuple[int, int], list[int]] = {}

    for a, b, c in faces:
        e1 = (a, b) if a < b else (b, a)
        e2 = (b, c) if b < c else (c, b)
        e3 = (a, c) if a < c else (c, a)

        edge_to_opp.setdefault(e1, []).append(c)
        edge_to_opp.setdefault(e2, []).append(a)
        edge_to_opp.setdefault(e3, []).append(b)

    return edge_to_opp


def find_ijlr_vertices(
        edge: np.ndarray,
        edge_to_opp: dict[tuple[int, int], list[int]]
) -> tuple[int, int]:
    """
    The rotation matrix Tk is given as a transformation that maps the vertices
    around the edge to new positions as closely as possible in a least-squares
    sense. WE SAMPLE FOUR VERTICES around the edge as a context to derive the
    local transformation Tk. It is possible to sample an arbitrary number of
    vertices greater than three here, but four is the most straightforward, and we
    have found that it produces good results. An exception applies to edges on
    the boundary. In those cases, we only USE THREE VERTICES to compute Tk.

    Source: Igarashi et al., 2009:

    l x--------x j
            // edge
        i x-------x r

    Find indices of vertex l and r. When edge i-j is situated at the edge of graph,
    then use only vertex l and r will be set as -1.

    :param edge: np.ndarray
    :param edge_to_opp: precomputed mapping from undirected edge to opposite vertices
    :return: (l, r)
    """
    i = int(edge[0])
    j = int(edge[1])
    key = (i, j) if i < j else (j, i)

    opp = edge_to_opp.get(key, [])
    if len(opp) == 0:
        return -1, -1
    if len(opp) == 1:
        return int(opp[0]), -1

    return int(opp[0]), int(opp[1])


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
