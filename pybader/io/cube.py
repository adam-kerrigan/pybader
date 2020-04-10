"""Module for handling Gaussian/CP2K style density files.
"""
import numpy as np
import os
from time import time
from ..utils import (
        tqdm_wrap,
        fortran_format,
        python_format,
)

__extensions__ = ['.cube']
__args__ = ['orbitals']

# Unit conversions
bohr_to_ang = .52917721067
ang_to_bohr = 1 / bohr_to_ang


def read(fn, orbitals=0):
    """Read the charge density from a cube file.

    Splits the density into blocks of buffer_size and parses it in chunks.
    Buffer_size must be positive and an integer. If decomposed to orbitals can
    either sum all for total charge density, sum a selection of orbitals or
    return all for user processing. Units are converted to VASP chgcar units.

    args:
        fn: name of the file to open.
        orbitals: how to deal with nval > 1, often associated with molecular
                  orbitals. Passing an iterator returns the sum of the passed
                  orbitals, passing an int > 0 returns just that orbital,
                  passing an int < 0 return the whole charge array in the order
                  density['charge'][nval,nx,ny,nz], the default value, 0,
                  returns a sum of all nvals unless atom number indicator is
                  positive,vin which case it returns just the first nval.
    return:
        density: dict containing 3d-arrays for charge and spin densities.
        lattice: 3x3 array with lattice vectors as rows.
        atoms: atomic positions in Cartesian basis.
        file_info: information about the file type and the write function.
    """
    t0 = time()
    density = dict()
    prefix, filename = os.path.split(fn)
    prefix = os.path.join(prefix, '')
    with open(fn, 'r') as f:
        print(f"  Reading {f.name} as cube format.")
        # first two lines are comments.
        _ = f.readline()
        _ = f.readline()
        line = f.readline().strip().split()
        atom_sum = int(line[0])
        origin = np.array(line[1:4], dtype=np.float64)
        if len(line) > 4:
            nval = int(line[5])
        else:
            nval = 1
        grid = np.zeros(3, dtype=np.int64)
        lattice = np.zeros((3, 3), dtype=np.float64)
        for i in range(3):
            line = f.readline().strip().split()
            grid[i] = line[0]
            lattice[i] = line[1:]
            lattice[i] *= grid[i]
        print(f"  {' x '.join(grid.astype(str))} grid size.")
        # read atomic positions and the blank line after
        atom_types = np.zeros(abs(atom_sum), dtype=np.int64)
        atoms = np.zeros((abs(atom_sum), 3), dtype=np.float64)
        for i in range(abs(atom_sum)):
            line = f.readline().strip().split()
            atom_types[i] = line[0]
            atoms[i] = line[-3:]
        # convert to fractional coordinates and then wrap in cell
        atoms = np.dot(atoms, np.linalg.inv(lattice))
        atoms %= 1
        # convert back
        atoms = np.dot(atoms, lattice)
        if atom_sum < 0:
            line = f.readline().strip().split()
            dset_ids = np.zeros(int(line.pop(0)), dtype=np.int64)
            nval = dset_ids.shape[0]
            count = 0
            while count < nval:
                for m in line:
                    dset_ids[count] = m
                    count += 1
                line = f.readline().strip().split()
        grid_pts = np.prod(grid)
        nx, ny, nz = grid
        record_pts = nz * nval
        grid_lines = record_pts // 6
        grid_mod = record_pts % 6
        # save the current file position and get the line length.
        charge_pos = f.tell()
        line_len = len(f.readline())
        f.seek(charge_pos)
        # set up buffer numbers.
        buffer_size = grid_lines
        buffer_range = [buffer_size]
        charge = np.zeros((nx, ny, nz * nval), dtype=np.float64)
        for x in tqdm_wrap(range(nx), desc="Charge density:"):
            for y in range(ny):
                idx = 0
                for buff in buffer_range:
                    # cube has 6 voxels per line.
                    idx_inc = buff * 6
                    buff_b = buff * line_len
                    line = f.read(buff_b).strip().split()
                    charge[x, y, idx:idx + idx_inc] = line
                    idx += idx_inc
                # get the last non-complete line.
                if grid_mod != 0:
                    line = f.readline().strip().split()
                    charge[x, y, -grid_mod:] = line
        print(f"  File {f.name} closed. ", end='')
    if nval > 1:
        # how are we handling nval > 1
        if hasattr(orbitals, '__iter__'):
            # sum all given orbitals
            charge = charge.reshape(nx, ny, nz, nval)
            charge = np.swapaxes(charge, 0, -1)
            density['charge'] = np.sum([charge[dset_ids.index(int(m))]
                    for m in orbitals], axis=0)
        elif orbitals < 0:
            # return the entire file
            charge = charge.reshape(nx, ny, nz, nval)
            density['charge'] = np.swapaxes(charge, 0, -1)
        elif orbitals > 0:
            # return specific orbital
            charge = charge.reshape(nx, ny, nz, nval)
            charge = np.swapaxes(charge, 0, -1)
            density['charge'] = charge[dset_ids.index(int(orbitals))].copy()
        elif atom_sum > 0:
            # return just first value (useful for gradient cubes)
            charge = charge.reshape(nx, ny, nz, nval)
            density['charge'] = np.swapaxes(charge, 0, -1)[0].copy()
        else:
            # sum all nvals
            charge = charge.reshape(nx, ny, nz, nval)
            charge = np.swapaxes(charge, 0, -1)
            density['charge'] = np.sum(charge, axis=0)
        del charge
    else:
        density['charge'] = charge
    print(f"Time taken: {time() - t0:0.3f}s", end='\n\n')
    lattice *= bohr_to_ang
    atoms *= bohr_to_ang
    lat_vol = np.abs(np.dot(lattice[0], np.cross(*lattice[1:])))
    density['charge'] *= lat_vol / bohr_to_ang**3
    file_info = {
            'filename': fn,
            'prefix': prefix,
            'file_type': 'cube',
            'write_function': write,
            'elements': atom_types,
            'voxel_offset': np.array([.5, .5, .5])
    }
    return density, lattice, atoms, file_info


def write(fn, atoms, lattice, density, file_info, prefix=None, suffix='.cube'):
    """Write a cube style charge density

    args:
        fn: filename
        atoms: the atoms for the structure
        lattice: lattice defining cell
        file_info: dictionary containing everything from file_info exported by
                   read function plus optional fortran_format flag
        prefix: string to be placed infront of filename
        suffix: string to be placed at end of filename
    """
    if prefix is not None:
        fn = prefix + fn
    fn += suffix
    if file_info.get('fortran_format', 0) == 2:
        output_format = fortran_format
    elif file_info.get('fortran_format', 0) == 1:
        def output_format(a, p):
            return python_format(a, p, ' ')
    else:
        output_format = python_format
    charge = density['charge']
    # convert to bohr
    atoms *= ang_to_bohr
    lat_vol = np.abs(np.dot(lattice[0], np.cross(*lattice[1:])))
    charge *= ang_to_bohr**3 / lat_vol
    lattice *= ang_to_bohr
    lattice /= charge.shape

    buffer_size = charge.shape[2] // 6
    buffer_remainder = charge.shape[2] % 6
    buffer_flag = buffer_remainder != 0

    lattice_width = np.max(np.log10(np.abs(lattice[lattice != 0]))) + 9
    lattice_width = max([int(lattice_width), 9]) + 1
    lattice_prec = 17 - lattice_width
    atoms_width = np.max(np.log10(np.abs(atoms[atoms != 0]))) + 9
    atoms_width = max([int(atoms_width), 9]) + 1
    atoms_prec = 17 - atoms_width
    with open(fn, 'w') as f:
        f.write("Cube File writen in pybader\n")
        f.write(file_info['comment'])
        f.write(f"{atoms.shape[0]:>5}{'  0.0000000'*3}\n")
        for i, lat in enumerate(lattice):
            x, y, z = lat
            f.write(f"{charge.shape[i]:>5}")
            f.write(f" {x:> {10}.{lattice_prec}f}")
            f.write(f" {y:> {10}.{lattice_prec}f}")
            f.write(f" {z:> {10}.{lattice_prec}f}\n")
        for i, atom in enumerate(atoms):
            x, y, z = atom
            f.write(f"{file_info['elements'][i]:>5}")
            f.write('  0.0000000')
            f.write(f" {x:> {10}.{atoms_prec}f}")
            f.write(f" {y:> {10}.{atoms_prec}f}")
            f.write(f" {z:> {10}.{atoms_prec}f}\n")
        for i in tqdm_wrap(range(charge.shape[0]), desc=f"{fn}:"):
            for j in range(charge.shape[1]):
                r = charge[i, j][:buffer_size * 6].reshape((buffer_size, 6))
                out = output_format(r, 5)
                if buffer_flag:
                    out += output_format([charge[i, j][-buffer_rem:]], 5)
                f.write(out)
