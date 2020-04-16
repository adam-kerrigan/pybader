"""Misc utility functions.
"""
import numpy as np
from numba import njit, types
from numba.typed import Dict
from time import sleep
from tqdm import tqdm
from shutil import get_terminal_size
from io import StringIO
from contextlib import contextmanager
import sys


def fortran_format(a, prec):
    """Fortmat an array into 'Fortran' standard form and return as string.

    All numbers are shown shifted one decimal place to the right and negative
    numbers have leading zero replaced with a minus sign.

    args:
        a: the array
        prec: the precision to quote elements of array to
    """
    out_shape = (a.shape[0], a.shape[1] * 5)
    a = a.flatten()
    a = a.reshape((a.shape[0], 1))
    sign = np.zeros(a.shape, dtype='<U3')
    val = np.zeros(a.shape, dtype=f'<U{prec}')
    E = np.zeros(a.shape, dtype='<U2')
    exp0 = np.zeros(a.shape, dtype='<U1')
    exp1 = np.zeros(a.shape, dtype='<U2')
    new_line = np.zeros((out_shape[0], 1), dtype='<U1')
    new_line[:] = '\n'
    abs_a = np.abs(a)
    exp = np.zeros(a.shape, dtype=np.int64)
    abs_exp = np.zeros(a.shape, dtype=np.int64)
    value = np.zeros(a.shape, dtype=np.int64)
    a_ne_0 = np.where(a != 0)
    a_l_0 = np.where(a < 0)
    exp[a_ne_0] = np.floor(np.log10(abs_a[a_ne_0])) + 1
    abs_exp[:] = np.abs(exp)
    value[a_ne_0] = 0.5 + abs_a[a_ne_0] / np.power(10.0, exp[a_ne_0] - prec)
    exp_l_0 = np.where(exp < 0)
    exp_l_10 = np.where(abs_exp < 10)
    sign[:] = ' 0.'
    sign[a_l_0] = ' -.'
    val[:] = '0' * prec
    val[a_ne_0] = value[a_ne_0]
    E[:] = 'E+'
    E[exp_l_0] = 'E-'
    exp0[exp_l_10] = '0'
    exp1[:] = '0'
    exp1[a_ne_0] = abs_exp[a_ne_0]
    out = np.concatenate((sign, val, E, exp0, exp1), axis=1).reshape(out_shape)
    out = np.concatenate((out, new_line), axis=1).flatten()
    return ''.join(out)


def python_format(a, prec, align=''):
    """Format an array into standard form and return as a string.

    args:
        a: the array
        prec: the precision to output the array to
        align: anything that can go before the . in python formatting
    """
    format_ = (f' {{:{align}.{prec}E}}' * a.shape[1] + '\n') * a.shape[0]
    return format_.format(*a.flatten())


@contextmanager
def nostdout():
    """Redirect stdout.
    """
    save_stdout = sys.stdout
    sys.stdout = StringIO()
    yield
    sys.stdout = save_stdout


def progress_bar_update(bar, progress, rate=0.05):
    """Updates a manual progress bar.

    args:
        bar: a tqdm (wrapped) progress bar
        progress: shared array storing the current progress as first element
        rate: rate at which to check the current progress
    """
    while True:
        bar.update(progress[0] - bar.n)
        if bar.n == bar.total:
            bar.close()
            break
        sleep(rate)


def tqdm_wrap(*args, **kwargs):
    """Wrapper for progress bar.

    Formats the bar nicely and sets a fixed width if the output width is wider
    than 80 characters.
    """
    ncols, _ = get_terminal_size((0, 0))
    bar_format = "  {desc} [{bar}] {percentage:3.0f}% {elapsed}<{remaining}  "
    if ncols < 80:
        ncols = None
    else:
        ncols = 80
    kwargs_update = {
            **kwargs,
            'ascii': True,
            'ncols': ncols,
            'bar_format': bar_format,
            'file': sys.stdout,
    }
    return tqdm(*args, **kwargs_update)


@njit(cache=True, nogil=True)
def array_assign(old_array, old_array_len, array_len):
    """Allocate a new length to the array.

    Reallocate the array to either increase the size or set the correct length
    to Nx3 arrays.

    args:
        old_array: the over or under sized array.
        old_array_len: the old size of the array.
        array_len: the desired length of the array.

    returns:
        array
    """
    array = np.zeros((array_len, 3), dtype=old_array.dtype)
    for i in range(old_array_len):
        array[i] = old_array[i]
    return array


@njit(cache=True, nogil=True)
def array_merge(a, b):
    """Merge two arrays

    args:
        a: array length la
        b: array length lb
    returns:
        array length la + lb
    """
    out = np.zeros((a.shape[0] + b.shape[0], 3), dtype=b.dtype)
    for i in range(out.shape[0]):
        if i < a.shape[0]:
            out[i] = a[i]
        else:
            out[i] = b[i - a.shape[0]]
    return out


@njit(cache=True, nogil=True)
def atom_assign(bader_max, atoms, lattice, i_c):
    """Assign Bader volumes to atoms.

    args:
        bader_max: positions of maxima in cartesian
        atoms: position atoms in cartesian
        lattice: the lattice of the cell for periodic boundary conditions
        i_c: an index counter for progress checking
    returns:
        bader_atom: list linking bader volumes (index) to atom number
        bader_distance: list linking bader volumes (index) to distance from atom
    """
    pbc_l = np.zeros((3, 3), dtype=np.float64)
    pbc = np.zeros(3, dtype=np.float64)
    assigned_atom = np.zeros(bader_max.shape[0], dtype=np.int64)
    assigned_distance = np.zeros(bader_max.shape[0], dtype=np.float64)
    atom = np.zeros(3, dtype=np.float64)
    for i in range(bader_max.shape[0]):
        i_c[0] += 1
        b_max = bader_max[i]
        min_distance = ((b_max[0] - (atoms[0][0] + pbc[0]))**2
                      + (b_max[1] - (atoms[0][1] + pbc[1]))**2
                      + (b_max[2] - (atoms[0][2] + pbc[2]))**2)
        atom_num = 0
        for j in range(atoms.shape[0]):
            for k in range(3):
                atom[k] = atoms[j][k]
            for x in range(-1, 2):
                for k in range(3):
                    pbc_l[0, k] = lattice[0, k] * x
                for y in range(-1, 2):
                    for k in range(3):
                        pbc_l[1, k] = lattice[1, k] * y
                    for z in range(-1, 2):
                        for k in range(3):
                            pbc_l[2, k] = lattice[2, k] * z
                        for k in range(3):
                            pbc[k] = pbc_l[0, k] + pbc_l[1, k] + pbc_l[2, k]
                        dist = ((b_max[0] - (atom[0] + pbc[0]))**2
                              + (b_max[1] - (atom[1] + pbc[1]))**2
                              + (b_max[2] - (atom[2] + pbc[2]))**2)
                        if dist < min_distance:
                            min_distance = dist
                            atom_num = j
        assigned_atom[i] = atom_num
        assigned_distance[i] = min_distance**.5
    return assigned_atom, assigned_distance


@njit(cache=True, nogil=True)
def charge_sum(charge, volume, voxel_volume, density, volumes):
    """Sum the charge of voxels with the same vol_num.

    args:
        charge: output array for charge
        volume: output array for the volume
        voxel_volume: float indicating volume of a voxel
        density: a reference density for charge summation
        volumes: a voxel to vol_num map
    """
    npts = density.shape[0] * density.shape[1] * density.shape[2]
    for i in np.ndindex(volumes.shape):
        atom_num = volumes[i]
        charge[atom_num] += density[i] / npts
        volume[atom_num] += voxel_volume


def dtype_calc(max_val):
    """Returns the dtype required to display the max val.

    args:
        max_val: set to negative to return signed int
    """
    dtype_list = [
        ['int8', 'int16', 'int32', 'int64'],
        ['uint8', 'uint16', 'uint32', 'uint64'],
    ]
    if max_val < 0:
        max_val *= -2
        dtype = dtype_list[0]
    else:
        dtype = dtype_list[1]
    if max_val <= 255:
        return dtype[0]
    elif max_val <= 65535:
        return dtype[1]
    elif max_val <= 4294967295:
        return dtype[2]
    else:
        return dtype[3]


@njit(cache=True)
def dtype_change(a, out):
    for i in np.ndindex(a.shape):
        out[i] = a[i]
    return out


@njit(cache=True, nogil=True)
def edge_assign(edges, volumes):
    """Resolve the edge crossing in threaded Bader methods.

    args:
        edges: array of Bader maximia assigned out of volume scope
        volumes: array of voxel to vol_nums map
    """
    swap = Dict.empty(
        key_type=types.int64,
        value_type=types.int64
    )
    for i in np.ndindex(volumes.shape):
        vol_num = volumes[i]
        if vol_num < -2:
            if vol_num not in swap:
                p = edges[-1 * (vol_num + 3)]
                swap[vol_num] = volumes[p[0], p[1], p[2]]
            volumes[i] = swap[vol_num]


@njit(cache=True)
def factor_3d(x):
    """Find 3 factors of a number for splitting volumes for threading.

    args:
        x: number to be factorised
    returns:
        tuple of factors
    """
    fac = list()
    fac.append((1, x))
    for i in range(2, x):
        if i >= (x / fac[-1][0]):
            break
        elif x % i == 0:
            fac.append((i, x // i))
    split = fac[-1]
    fac = list()
    fac.append((1, split[1]))
    for i in range(2, split[1]):
        if i >= (split[1] / fac[-1][0]):
            break
        elif split[1] % i == 0:
            fac.append((i, split[1] // i))
    out = (fac[-1][1], fac[-1][0], split[0])
    if fac[-1][0] == 1:
        fac.pop(0)
        fac.append((1, split[0]))
        for i in range(2, split[0]):
            if i >= (split[0] / fac[-1][0]):
                break
            elif split[0] % i == 0:
                fac.append((i, split[0] // i))
        out = (fac[-1][1], fac[-1][0], split[1])
    return out


@njit(cache=True, nogil=True)
def surface_dist(idx, shape, known, volumes, lattice, atoms, i_c):
    """Calculate the minimum distance to the surface of the Bader volume.

    args:
        idx: position in volume space
        shape: length to iterate over
        known: edge indicators
        lattice: the lattice of the cell for periodic boundary conditions
        atoms: position to measure distance to
        i_c: an index counter for progress checking
    returns:
        distance: list linking atom volumes (index) to distance from atom
    """
    pbc_l = np.zeros((3, 3), dtype=np.float64)
    pbc = np.zeros(3, dtype=np.float64)
    distance = np.zeros(atoms.shape[0], dtype=np.float64)
    max_distance = np.zeros(atoms.shape[0], dtype=np.float64)
    atom = np.zeros(3, dtype=np.float64)
    p = np.zeros(3, dtype=np.int64)
    pc = np.zeros(3, dtype=np.float64)
    for j in range(atoms.shape[0]):
        max_distance[j] = known.shape[0]**2 + known.shape[1]**2 + known.shape[2]**2
    for nx in range(idx[0], idx[0] + shape[0]):
        p[0] = nx
        for ny in range(idx[1], idx[1] + shape[1]):
            p[1] = ny
            for nz in range(idx[2], idx[2] + shape[2]):
                p[2] = nz
                if known[nx, ny, nz] != -2:
                    continue
                i_c[0] += 1
                vol_num = volumes[nx, ny, nz]
                min_distance = max_distance[vol_num]
                for j in range(3):
                    atom[j] = atoms[vol_num][j]
                    pc[j] = lattice[0, j] * p[0] / volumes.shape[0]
                    pc[j] += lattice[1, j] * p[1] / volumes.shape[1]
                    pc[j] += lattice[2, j] * p[2] / volumes.shape[2]
                for x in range(-1, 2):
                    for j in range(3):
                        pbc_l[0, j] = lattice[0, j] * x
                    for y in range(-1, 2):
                        for j in range(3):
                            pbc_l[1, j] = lattice[1, j] * y
                        for z in range(-1, 2):
                            for j in range(3):
                                pbc_l[2, j] = lattice[2, j] * z
                            for j in range(3):
                                pbc[j] = pbc_l[0, j] + pbc_l[1, j] + pbc_l[2, j]
                            dist = ((pc[0] - (atom[0] + pbc[0]))**2
                                  + (pc[1] - (atom[1] + pbc[1]))**2
                                  + (pc[2] - (atom[2] + pbc[2]))**2)
                            if dist < min_distance:
                                min_distance = dist
                distance[vol_num] = min_distance**.5
                max_distance[vol_num] = min_distance
    return distance


@njit(cache=True)
def vacuum_assign(density, volumes, vac_tol):
    """Create a voxel to vol_num map with masked values below a threshold.

    args:
        density: the density to be used as a reference
        volumes: an empty array to be filled
        vac_tol: tolerance of what to consider vacuum
    """
    for i in np.ndindex(volumes.shape):
        if density[i] <= vac_tol:
            volumes[i] = -1
    return volumes


@njit(cache=True, nogil=True)
def volume_assign(volumes, swap, i_c):
    """Swap vol_nums to specific values.

    args:
        volumes: array of voxel to vol_nums map
        swap: array containing value to change vol_num (index) to
        i_c: index counter for progress bar
    """
    idx_c = 0
    for idx in np.ndindex(volumes.shape):
        if idx[0] != idx_c:
            idx_c += 1
            i_c[0] += 1
        vol_num = volumes[idx]
        if vol_num >= 0:
            volumes[idx] = swap[vol_num]
    i_c[0] += 1


@njit(cache=True, nogil=True)
def volume_extend(volumes, positive, new):
    """Expand the volumes array.

    Consider 1d array [0 1 2 3 4 -5 -4 -3 -2 -1] with positive length 5.
    expanding this array to new length 15 would look like
    [0 1 2 3 4 0 0 0 0 0 -5 -4 -3 -2 -1] so that values accessed by a negative
    index are still in the same place. This allows the array to remain as small
    as possible whilst covering the periodic repeats.

    args:
        volumes: array to be extended
        positive: array of the length of positive indices in each axis
        new: new length of each axis
    """
    i = np.zeros(3, dtype=np.int64)
    vx, vy, vz = volumes.shape
    out = np.zeros((new[0], new[1], new[2]), dtype=volumes.dtype)
    for ix in range(vx):
        if ix < positive[0]:
            i[0] = ix
        else:
            i[0] = ix - vx + new[0]
        for iy in range(vy):
            if iy < positive[1]:
                i[1] = iy
            else:
                i[1] = iy - vy + new[1]
            for iz in range(vz):
                if iz < positive[2]:
                    i[2] = iz
                else:
                    i[2] = iz - vz + new[2]
                out[i[0], i[1], i[2]] = volumes[ix, iy, iz]
    return out


@njit(cache=True, nogil=True)
def volume_mask(volumes, density, vol_num):
    """Mask a specific vol_num onto the density.

    args:
        volumes: array containing voxel to vol_num map
        density: the density to apply the mask to
        vol_num: the volume number to show the density for
    returns:
        density of specific vol_num
    """
    out = np.zeros(density.shape, dtype=np.float64)
    for idx in np.ndindex(volumes.shape):
        if vol_num == volumes[idx]:
            out[idx] = density[idx]
    return out


@njit(cache=True)
def volume_merge(volumes, vol, idx, shape):
    """Merge voxel to vol_num maps.

    args:
        volumes: array of voxel to vol_nums map
        vol: sub-array of volumes to be merged in
        idx: index at which to start the merge
        shape: the extent of what to merge
    """
    p = np.zeros(3, dtype=np.int64)
    for i in np.ndindex(shape[0], shape[1], shape[2]):
        for j in range(3):
            p[j] = idx[j] + i[j]
        volumes[p[0], p[1], p[2]] = vol[i]


@njit(cache=True, nogil=True)
def volume_offset(vol, bader, edge):
    """Offset the values in volumes array to acoount for completed threads.

    args:
        vols: sub-array of voxel to vol_nums map
        bader: the current amount of Bader maxima
        edge: the current amount of outside volumes space maxima
    """
    for i in np.ndindex(vol.shape):
        if vol[i] > 0:
            # -1 means 0 is now first Bader volume
            vol[i] += bader - 1
        elif vol[i] < -2:
            vol[i] -= edge
