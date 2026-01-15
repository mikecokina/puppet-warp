from __future__ import annotations

import re
from pathlib import Path

from pwarp import np
from pwarp.core import dtype


def read_wavefront(path: str | Path) -> tuple[int, int, np.ndarray, np.ndarray]:
    """Read a Wavefront OBJ file containing 2D vertices and triangular faces.

    Parse vertex (`v`) and face (`f`) records from an OBJ file. Treat vertices as
    2D by ignoring any third coordinate. Support common face formats such as
    `f v1 v2 v3`, `f v1/vt1 v2/vt2 v3/vt3`, `f v1//vn1 ...`, and `f v1/vt1/vn1 ...`.

    :param path: Path to the OBJ file.
    :return: Tuple of (num_vertices, num_faces, vertices, faces).
    """
    obj_path = Path(path)

    def parse_line(line_: str) -> np.ndarray:
        # If line starts with `f` it means, face is defined and
        # we wanna be able to hande different wavefront forms.
        # Supported wavefront forms:
        #   f v1 v2 v3 ....
        #   f v1/vt1 v2/vt2 v3/vt3 ...
        #   f v1/vt1/vn1 v2/vt2/vn2 v3/vt3/vn3 ...
        #   f v1//vn1 v2//vn2 v3//vn3 ...
        if line_.startswith("f"):
            parts = line_.split()
            indices = [int(c.split("/")[0]) for c in parts[1:]]
            return np.array(indices, dtype=dtype.INDEX)

        if line_.startswith("v"):
            numbers = re.findall(r"[-+]?\d*\.*\d+", line_)
            return np.array(numbers, dtype=dtype.FLOAT)

        return np.array([], dtype=dtype.FLOAT)

    vertices: list[np.ndarray] = []
    faces: list[np.ndarray] = []

    for line in obj_path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        if line[0] not in {"v", "f"}:
            # Skip all lines different from vertex/face definition.
            continue

        data = parse_line(line)
        if line[0] == "v":
            # Append vertex (expeceting 2D space, so we ingore 3rd cooridnate).
            vertices.append(np.array(data[:2], dtype=dtype.FLOAT))
        elif line[0] == "f":
            # Append face.
            faces.append(np.array(data[:3], dtype=dtype.INDEX))

    vertices_arr = np.array(vertices, dtype=dtype.FLOAT)
    # Shift faces by 1 due to convinience in wavefront obj.
    faces_arr = np.array(faces, dtype=dtype.INDEX) - 1

    return len(vertices_arr), len(faces_arr), vertices_arr, faces_arr


def save_wavefront(
    path: str | Path,
    no_vertices: int,
    no_faces: int,
    vertices: np.ndarray,
    faces: np.ndarray,
) -> None:
    """Save vertices and faces to a Wavefront OBJ file.

    Write vertices as 2D points with a zero Z coordinate. Write faces as 1-based
    indices as required by the OBJ format.

    :param path: Output OBJ file path.
    :param no_vertices: Number of vertices to write.
    :param no_faces: Number of faces to write.
    :param vertices: Vertex array of shape (N, 2).
    :param faces: Face index array of shape (M, 3), zero-based.
    """
    out_path = Path(path)
    faces_1based = faces + 1

    lines: list[str] = [
        f"#vertices: {no_vertices}\n",
        f"#faces: {no_faces}\n",
    ]

    lines.extend(
        f"v {v[0]} {v[1]} 0\n"
        for v in vertices
    )

    lines.extend(
        f"f {int(face[0])} {int(face[1])} {int(face[2])}\n"
        for face in faces_1based
    )

    out_path.write_text("".join(lines), encoding="utf-8")
