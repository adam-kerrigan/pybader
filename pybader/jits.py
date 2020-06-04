"""This module contains sample arguments for the jitable functions.
"""
from itertools import product

import numpy as np

from pybader import methods, refinement, utils

ints = [np.int8, np.int16, np.int32, np.int64]
uints = [np.uint8, np.uint16, np.uint32, np.uint64]
floats = [np.float32, np.float64]
d2 = (3, 3)
d3 = (3, 3, 3)


def array(dim, dtype=np.float64, val=0):
    array = np.zeros(dim, dtype=dtype, order='C')
    if val != 0:
        array[:] = val
    return array


method_sig = [(array(d3), array(d3, dtype=I, val=-1), array(3, dtype=ints[-1]),
               array(d3), array(d2), array(1, dtype=ints[-1])) for I in ints]

methods_jit = {key: method_sig for key in methods.__contains__}

method_sig = [(array(d3, dtype=ints[0]), array(d3, dtype=ints[0], val=2),
               array(d3), array(d3, dtype=I), array(
                   3, dtype=ints[-1]), array(d3),
               array(d2), array(1, dtype=ints[-1])) for I in ints]
edge_sig = [(array(d3, dtype=ints[0]), array(d3), array(d3, dtype=I))
            for I in ints]

refinement_jit = {
    **{key: method_sig for key in refinement.__contains__},
    **{key: edge_sig for key in ['edge_find', 'edge_check']}
}

utils_jit = {
    'array_assign': [
        (array(d2, dtype=ints[-1]), ints[-1](0), ints[-1](0)),
        (array(d2, dtype=uints[-1]), ints[-1](0), ints[-1](0)),
    ],
    'array_merge': [
        (array(d2, dtype=ints[-1]), array(d2, dtype=ints[-1]))
    ],
    'atom_assign': [
        (array(d2), array(d2), array(d2), array(1, dtype=ints[-1])),
    ],
    'charge_sum': [
        *[(array(1), array(1), floats[-1](0), array(d3), array(d3, dtype=I))
          for I in ints],
    ],
    'dtype_change': [
        *[(array(d3, dtype=I), array(d3, dtype=II))
          for I, II in product(ints, ints)],
        *[(array(d3, dtype=I), array(d3, dtype=uI))
          for I, uI in product(ints, uints)],
    ],
    'edge_assign': [
        *[(array(d2, dtype=ints[-1]), array(d3, dtype=I)) for I in ints],
    ],
    'factor_3d': [
        (ints[-1](0),),
    ],
    'surface_dist': [
        *[(array(3, dtype=ints[-1]), array(3, dtype=ints[-1], val=3),
           array(d3, dtype=ints[0]), array(d3, dtype=I), array(d2),
           array(d2), array(1, dtype=ints[-1])) for I in ints],
    ],
    'vacuum_assign': [
        *[(array(d3), array(d3, dtype=I), floats[-1](0), array(d3),
           floats[-1](0)) for I in ints],
    ],
    'volume_assign': [
        *[(array(d3, dtype=I, val=-1), array(1, dtype=ints[-1]),
           array(1, dtype=ints[-1])) for I in ints],
    ],
    'volume_extend': [
        *[(array(d3, dtype=I), array(3, dtype=ints[-1]),
           array(3, dtype=ints[-1])) for I in ints],
    ],
    'volume_mask': [
        *[(array(d3, dtype=I), array(d3), ints[-1](0)) for I in ints],
    ],
    'volume_merge': [
        *[(array(d3, dtype=I), array(d3, dtype=I), array(3, dtype=ints[-1]),
           array(3, dtype=ints[-1])) for I in ints],
    ],
    'volume_offset': [
        *[(array(d3, dtype=I), ints[-1](0), ints[-1](0))
          for I in ints],
    ],
}

jit_functs = {
    'methods': methods_jit,
    'refinement': refinement_jit,
    'utils': utils_jit,
}
