from pathlib import Path
from unittest import TestCase
from parameterized import parameterized

from numpy.testing import assert_array_equal

# noinspection PyProtectedMember
from pwarp._io import read_wavefront


class DataTestCase(TestCase):
    DATA_DIR = Path(__file__).parent / 'data'


class DataReadTestCase(DataTestCase):
    VERTICES = [[1, 2], [4, 5], [7, 8]]
    FACES = [[0, 1, 2]]

    @parameterized.expand([
        ['simple.obj'],
        ['texture.obj'],
        ['normal.obj'],
        ['texnorm.obj']
    ])
    def test_read_wavefront(self, file_: str):
        fpath = self.DATA_DIR / file_
        nv, nf, v, f = read_wavefront(str(fpath))

        assert nv == len(self.VERTICES)
        assert nf == len(self.FACES)
        assert_array_equal(self.FACES, f)
        assert_array_equal(self.VERTICES, v)
