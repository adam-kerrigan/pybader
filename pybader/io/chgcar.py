"""Module for handling VASP style density files.
"""
import numpy as np
import os
from time import time
from ..utils import (
        tqdm_wrap,
        fortran_format,
        python_format,
)
from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
)

__extensions__ = ['chgcar', '.vasp']
__args__ = ['charge_flag', 'spin_flag', 'buffer_size']


def read(fn, charge_flag=True, spin_flag=False, buffer_size=64):
    """Read the charge and/or spin density from a VASP chgcar.

    Splits the density into blocks of buffer_size and parses it in chunks. Can
    read just the density, just the spin or both. Buffer_size must be positive,
    an integer and greater than the number of grid points // 5. Ignores all
    augmentation charges.

    args:
        fn: name of the file to open.
        charge_flag: whether to read the charge density.
        spin_flag: whether to read the spin density.
        buffer_size: amount of lines to be read at once.

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
        print(f"  Reading {f.name} as CHGCAR format.")
        # fisrt line is comment from poscar.
        _ = f.readline()
        scale = np.array(f.readline().strip().split(), dtype=np.float64)
        lattice = np.zeros((3, 3), dtype=np.float64)
        for i in range(3):
            lattice[i] = f.readline().strip().split()
        atom_types = f.readline().strip().split()
        try:
            # check for atomic symbols line by assuming it's not there.
            atom_nums = np.array(atom_types, dtype=np.int64)
            atom_types = None
        except ValueError:
            atom_nums = np.array(f.readline().strip().split(), dtype=np.int64)
        atom_sum = atom_nums.sum()
        coord_system = f.readline().lstrip().lower()
        # read atomic positions and the blank line after
        atoms = np.zeros((atom_sum, 3), dtype=np.float64)
        for i in range(atom_sum):
            atoms[i] = f.readline().strip().split()
        # wrap all atoms inside the cell
        if coord_system[0] == 'd':
            atoms %= 1
        else:
            atoms = np.dot(atoms, np.linalg.inv(lattice))
            atoms %= 1
        _ = f.readline()
        grid = np.array(f.readline().strip().split(), dtype=np.int64)
        grid_pts = np.prod(grid)
        print(f"  {' x '.join(grid.astype(str))} grid size.")
        grid_lines = grid_pts // 5
        grid_mod = grid_pts % 5
        # save the current file position and get the line length.
        charge_pos = f.tell()
        line_len = len(f.readline())
        f.seek(charge_pos)
        # set up buffer numbers.
        if grid_lines > buffer_size:
            buffer_num = (grid_lines // buffer_size) + 1
            buffer_range = [buffer_size
                    if i < buffer_num - 1
                    else (grid_lines - (i * buffer_size))
                    for i in range(buffer_num)]
        else:
            # buffer larger than the amount of lines in density.
            buffer_size = grid_lines
            buffer_range = [buffer_size]
        if charge_flag:
            t1 = time()
            charge = np.zeros(grid_pts, dtype=np.float64)
            idx = 0
            for buff in tqdm_wrap(buffer_range, desc="Charge density:"):
                # chgcar has 5 voxels per line.
                idx_inc = buff * 5
                buff_b = buff * line_len
                charge[idx:idx+idx_inc] = f.read(buff_b).strip().split()
                idx += idx_inc
            # get the last non-complete line.
            if grid_mod != 0:
                charge[-grid_mod:] = f.readline().strip().split()
            charge = np.swapaxes(charge.reshape(grid[::-1]), 0, -1)
            density['charge'] = charge.copy()
            del charge
        if spin_flag:
            # goto the EOF and use this to find the mid-point.
            f.seek(0, 2)
            end_file = f.tell()
            # minus the line length as a measure againt being in the middle
            # of the grid-point line.
            spin_pos = int(.5 * (charge_pos + end_file) - line_len)
            if spin_pos < (charge_pos + grid_lines * line_len) * .75:
                print(f"  No spin density in {fn}")
            f.seek(spin_pos)
            _ = f.readline()
            while True:
                line = f.readline().strip().split()
                # int can't deal with standard form from a string, float can.
                if all(int(float(x)) == y for x, y in zip(line, grid)):
                    break
            # initialise spin density array and read in data.
            spin = np.zeros(grid_pts, dtype=np.float64)
            idx = 0
            for buff in tqdm_wrap(buffer_range, desc="Spin density:  "):
                # chgcar has 5 voxels per line.
                idx_inc = buff * 5
                buff_b = buff * line_len
                spin[idx:idx + idx_inc] = f.read(buff_b).strip().split()
                idx += idx_inc
            if grid_mod != 0:
                spin[-grid_mod:] = f.readline().strip().split()
            spin = np.swapaxes(spin.reshape(grid[::-1]), 0, -1)
            density['spin'] = spin.copy()
            del spin
        print(f"  File {f.name} closed. ", end='')
    # multiply lattice by the scaling factor and transpose in to column matrix.
    if scale.shape[0] == 1:
        lattice *= scale[0]
    else:
        for i in range(3):
            lattice[i] *= scale[i]
    # put atoms in Cartesian basis.
    atoms = np.dot(atoms, lattice)
    print(f"Time taken: {time() - t0:0.3f}s", end='\n\n')
    file_info = {
            'filename': filename,
            'prefix': prefix,
            'file_type': 'CHGCAR',
            'buffer_size': buffer_size,
            'write_function': write,
            'element_nums': atom_nums,
            'charge_flag': charge_flag,
            'spin_flag': spin_flag,
            'voxel_offset': np.zeros(3)
    }
    if atom_types is not None:
        file_info['elements'] = atom_types
    return density, lattice, atoms, file_info


def write(fn, atoms, lattice, density, file_info, prefix='', suffix='-CHGCAR'):
    """Write a VASP style charge density

    args:
        fn: filename
        atoms: the atoms for the structure
        lattice: lattice defining cell
        file_info: dictionary containing everything from file_info exported by
                   read function plus optional fortran_format flag
        prefix: string to be placed infront of filename
        suffix: string to be placed at end of filename
    """
    fn = prefix + fn + suffix
    if file_info.get('fortran_format', 0) == 2:
        output_format = fortran_format
    elif file_info.get('fortran_format', 0) == 1:
        def output_format(a, p):
            return python_format(a, p, ' ')
    else:
        output_format = python_format
    buffer_size = file_info['buffer_size']
    if file_info['charge_flag']:
        charge = density.get('charge')
        grid = np.prod(charge.shape)
        shape = charge.shape
        charge = np.swapaxes(charge, 0, -1).flatten()
        lines = grid // 5
        lines_rem = grid % 5
        lines_flag = lines_rem != 0
        buffer_range = lines // buffer_size
        if lines_flag:
            last_charge = output_format(np.array([charge[-lines_rem:]]), 11)
        charge = np.array_split(
            charge[:-lines_rem].reshape((lines, 5)),
            buffer_range
        )
    if file_info['spin_flag']:
        spin = density.get('spin')
        grid = np.prod(spin.shape)
        shape = spin.shape
        spin = np.swapaxes(spin, 0, -1).flatten()
        lines = grid // 5
        lines_rem = grid % 5
        lines_flag = lines_rem != 0
        buffer_range = lines // buffer_size
        if lines_flag:
            last_spin = output_format(np.array([spin[-lines_rem:]]), 11)
        spin = np.array_split(
            spin[:-lines_rem].reshape((lines, 5)),
            buffer_range
        )

    lattice_width = np.max(np.log10(np.abs(lattice[lattice != 0]))) + 9
    lattice_width = max([int(lattice_width), 9]) + 1
    lattice_prec = 17 - lattice_width
    atoms_width = np.max(np.log10(np.abs(atoms))).astype(int) + 9
    atoms_width = max([atoms_width, 9]) + 1
    atoms_prec = 17 - atoms_width
    with open(fn, 'w') as f:
        f.write(file_info['comment'])
        f.write(f"{1:0< 10.7f}\n")
        for x, y, z in lattice:
            f.write(f" {x:> {10}.{lattice_prec}f}")
            f.write(f" {y:> {10}.{lattice_prec}f}")
            f.write(f" {z:> {10}.{lattice_prec}f}\n")
        if file_info.get('elements', None) is not None:
            f.write('  '.join(file_info['elements'])+'\n')
        f.write('  '.join(file_info['element_nums'].astype(str))+'\n')
        f.write('Cartesian\n')
        for x, y, z in atoms:
            f.write(f" {x:> {10}.{atoms_prec}f}")
            f.write(f" {y:> {10}.{atoms_prec}f}")
            f.write(f" {z:> {10}.{atoms_prec}f}\n")
        x, y, z = shape
        f.write('\n')
        if file_info['charge_flag']:
            f.write(f" {x:>5} {y:>5} {z:>5}\n")
            for array in tqdm_wrap(charge, desc=f"{fn}:"):
                chunk = output_format(array, 11)
                f.write(chunk)
            if lines_flag:
                f.write(last_charge)
        if file_info['spin_flag']:
            f.write(f" {x:>5} {y:>5} {z:>5}\n")
            for array in tqdm_wrap(spin, desc=f"{fn}:"):
                chunk = output_format(array, 11)
                f.write(chunk)
            if lines_flag:
                f.write(last_spin)
