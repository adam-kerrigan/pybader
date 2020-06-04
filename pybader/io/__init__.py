"""Package for I/O handling.

All modules must have a __args__ and __ext__ variable set for determining what
arguments the read function takes and what extenstion to match in the filename.
Other information (eg. flags for the writting) should be set in the dictionary
file_info with filename, prefix, file_type, write_function and voxel_offset.
"""
from pybader.io import cube, pymatgen, vasp

try:
    from pybader.io import gpaw
except ImportError:
    pass
