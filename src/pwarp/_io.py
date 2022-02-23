import os
import re

from pwarp import np
from pwarp.core import dtype


def read_wavefront(path: str):
    def parse_line(_line):
        _line = re.findall(r"[-+]?\d*\.*\d+", _line)
        return np.array(_line, dtype=dtype.FLOAT)

    vertices, faces = [], []

    with open(path, "r") as file:
        for line in file:
            if line[0] == "#":
                # skip comments
                continue

            data = parse_line(line)
            if line[0] == "v":
                # append vertex
                vertices.append(np.array(data[:2], dtype=dtype.FLOAT))

            elif line[0] == "f":
                # append face
                faces.append(np.array(data[:3], dtype=dtype.INDEX))

    vertices = np.array(vertices, dtype=dtype.FLOAT)
    # shift faces by 1 due to convinience in wavefront obj
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
