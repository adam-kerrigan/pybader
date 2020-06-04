"""Module for handling GPAW input and output.

This module requires GPAW (https://wiki.fysik.dtu.dk/gpaw/) to run the read
function but is importable without it for use with ase. However it will not
show as an available filetype unless installed.
"""
import numpy as np

from .cube import write

try:
    from gpaw import restart
    GPAW_AVAIL = True
except ImportError:
    GPAW_AVAIL = False

__extensions__ = ['.gpw']
__args__ = ['gridref', 'spin_flag']


def read_obj(calc, gridref=4, spin_flag=False, fn='', prefix=''):
    """Read in from a GPAW/ase calculator object.

    args:
        calc: the calculator object
        gridref: the gridrefinemnet flag for get_all_electron_density method
        spin_flag: whether to read the spin density also
        fn: if this has come from a file, what file?
        prefix: if this has come from a file, where was it?
    """
    atoms_obj = calc.get_atoms()
    if calc.get_spin_polarized() and spin_flag:
        spin_0 = calc.get_all_electron_density(spin=0, gridrefinement=gridref)
        spin_1 = calc.get_all_electron_density(spin=1, gridrefinement=gridref)
        density_dict = {
            'charge': spin_0 + spin_1,
            'spin': spin_0 - spin_1,
        }
    else:
        density_dict = {
            'charge': calc.get_all_electron_density(gridrefinement=gridref)
        }
    lattice = np.zeros((3, 3), dtype=np.float64, order='C')
    lattice[:] = atoms_obj.cell[:]
    atoms = np.zeros(atoms_obj.positions.shape, dtype=np.float64, order='C')
    atoms[:] = atoms_obj.get_scaled_positions()
    atoms = np.dot(atoms, lattice)
    file_info = {
        'filename': fn,
        'prefix': prefix,
        'file_type': 'gpaw',
        'write_function': write,
        'elements': atoms_obj.get_atomic_numbers(),
        'voxel_offset': np.zeros(3),  # I am unsure on this
    }
    return density_dict, lattice, atoms, file_info


def read(fn, gridref=4, spin_flag=False):
    """Read in from a GPAW restart file, this function requires gpaw.
    args:
        fn: filename
        gridref: the gridrefinemnet flag for get_all_electron_density method
        spin_flag: whether to read the spin density also
    -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    Future: remove gpaw dependancy?
    """
    from gpaw import restart

    prefix, filename = os.path.split(fn)
    prefix = os.path.join(prefix, '')
    _, calc = restart(fn)
    return read_calc(calc, gridref, spin_flag, fn, prefix)
