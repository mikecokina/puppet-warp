import os
import re

from pwarp import np
from pwarp.core import dtype


def read_wavefront(path: str):
    def parse_line(_line: str):
        # If line starts with `f` it means, face is defined and
        # we wanna be able to hande different wavefront forms.
        # Supported wavefront forms:
        #   f v1 v2 v3 ....
        #   f v1/vt1 v2/vt2 v3/vt3 ...
        #   f v1/vt1/vn1 v2/vt2/vn2 v3/vt3/vn3 ...
        #   f v1//vn1 v2//vn2 v3//vn3 ...
        if _line.startswith('f'):
            return [int(c.split('/')[0]) for c in line.split(' ')[1:]]
        elif _line.startswith('v'):
            _line = re.findall(r"[-+]?\d*\.*\d+", _line)
        return np.array(_line, dtype=dtype.FLOAT)

    vertices, faces = [], []

    with open(path, "r") as file:
        for line in file:
            if line[0] not in ["v", "f"]:
                # Skip all lines different from vertex/face definition.
                continue

            data = parse_line(line)
            if line[0] == "v":
                # Append vertex (expeceting 2D space, so we ingore 3rd cooridnate).
                vertices.append(np.array(data[:2], dtype=dtype.FLOAT))

            elif line[0] == "f":
                # Append face.
                faces.append(np.array(data[:3], dtype=dtype.INDEX))

    vertices = np.array(vertices, dtype=dtype.FLOAT)
    # Shift faces by 1 due to convinience in wavefront obj.
    faces = np.array(faces, dtype=dtype.INDEX) - 1

    return len(vertices), len(faces), vertices, faces


def save_wavefront(path: str, no_vertices: int, no_faces: int, vertices: np.ndarray, faces: np.ndarray):
    __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    faces = faces[:] + 1
    with open(path, 'w') as f:
        f.write("#vertices: " + str(no_vertices) + "\n")
        f.write("#faces: " + str(no_faces) + "\n")
        for v in vertices:
            f.write("v " + str(v[0]) + " " + str(v[1]) + " 0\n")
        for face in faces:
            f.write("f " + str(int(face[0])) + " " + str(int(face[1])) + " " + str(int(face[2])) + "\n")
