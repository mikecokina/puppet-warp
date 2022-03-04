import os.path as op

from pwarp._np import np
from pwarp.conf import initialize_logger
from pwarp.data.puppet import PuppetObject
from pwarp.warp.warp import graph_defined_warp, graph_warp
from pwarp.mesh.tri import triangular_mesh
from pwarp.core.arap import StepOne as ARAPStepOne, StepTwo as ARAPStepTwo
from pwarp import _io
from pwarp.demo.demo import Demo


__all__ = (
    '__version__',
    '__author__',
    'np',
    'graph_defined_warp',
    'graph_warp',
    'triangular_mesh',
    'ARAPStepOne',
    'ARAPStepTwo',
    'PuppetObject',
    'get_default_puppet',
    'Demo'
)


initialize_logger()


def get_default_puppet():
    puppet_path = op.join(op.dirname(__file__), "data", "puppet.obj")
    no_r, no_f, r, f = _io.read_wavefront(puppet_path)
    return PuppetObject(vertices=r, faces=f, no_faces=no_f, no_vertices=no_r)


__version__ = "0.1"
__author__ = "mikecokina"
