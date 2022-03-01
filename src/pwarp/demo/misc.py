from pwarp import np


def is_close(x, y, vertices, _tol=4):
    vertices_scaled = vertices.copy().astype(int)
    close, index = False, -1

    for i in range(x - _tol, x + _tol):
        for j in range(y - _tol, y + _tol):
            if ([i, j] == vertices_scaled).all(axis=1).any():
                close = True
                click = np.array([i, j])
                index = np.where(np.all(click == vertices_scaled, axis=1))
                index = index[0][0]
                break
    if not close:
        index = -1

    return close, index
