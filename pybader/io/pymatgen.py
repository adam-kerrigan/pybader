"""Module for handling pymatgen VolumetricData objects.
"""
from itertools import groupby

import numpy as np

from .vasp import write

__extensions__ = None
__args__ = ['spin_flag']


def read_obj(obj, spin_flag=False):
    """Convert a VolumetricData object to input for Bader class.

    args:
        obj: VolumetricData object
        spin_flag: whether to read the spin density
    """
    density_dict = {}
    density_dict['charge'] = obj.data.get('total', None)
    if spin_flag:
        density_dict['spin'] = obj.data.get('diff', None)
    for key, value in density_dict:
        if value is None:
            density_dict.pop(key)
        else:
            density_dict[key] /= obj.structure.lattice.volume
    lattice = np.zeros((3, 3))
    atoms = np.zeros((len(obj.structure._sites), 3))
    lattice[:] = obj.structure.lattice.matrix
    atoms[:] = np.mod(obj.structure.frac_coords, 1)
    atoms = np.dot(atoms, lattice)
    atom_types = [site.specie.symbol for site in obj.structure.sites]
    atom_nums = [sym[0] for sym in groupby(atom_types)]
    file_info = {
        'filename': '',
        'prefix': '',
        'file_type': 'pymatgen object',
        'write_function': write,
        'elements': atom_types,
        'element_nums': atom_nums,
        'spin_flag': spin_flag,
        'voxel_offset': np.zeros(3)
    }
    return density_dict, lattice, atoms, file_info
