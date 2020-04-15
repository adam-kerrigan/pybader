"""Functions for handling the threading of intensive calculations.
"""
import numpy as np
from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
)
from pybader import refinement
from pybader import methods
from .utils import (
    array_merge,
    atom_assign,
    dtype_calc,
    dtype_change,
    edge_assign,
    factor_3d,
    progress_bar_update,
    surface_dist,
    tqdm_wrap,
    volume_assign,
    volume_merge,
    volume_offset,
)


def bader_calc(method, density, volumes, dist_mat, T_grad, threads):
    """Set up arrays for threading the Bader calculation method.

    args:
        method: name of method (name for name in methods.__contains__)
        density, volumes, dist_mat, T_grad: input for method
        threads: the number of threads to distribute over (Note: total number
                 will be threads + 1 as progress bar takes a thread)
    returns:
        array containing the postion of Bader maxima as voxel indices
    """
    bader = getattr(methods, method)
    thread = max([threads, 1])
    zipped = zip(density.shape, factor_3d(thread))
    split = np.array([x for _, x in sorted(zipped, key=lambda x: x[0])])
    vols = [volumes]
    for i in range(3):
        vc = vols.copy()
        vols = list()
        for v in vc:
            vt = np.array_split(v, split[i], axis=i)
            for val in vt:
                vols.append(np.ascontiguousarray(val))
    vol_s = np.array([v.shape for v in vols])
    idx = np.zeros((thread, 3), dtype=dtype_calc(max(density.shape)), order='C')
    c = -1
    k = (0, 0, 0)
    for i in np.ndindex(*split):
        j = np.where(np.subtract(i, k) != 0, 1, 0)
        ii = np.mod(np.add(idx[c], np.multiply(j, vol_s[c])), density.shape)
        c += 1
        idx[c] = ii
        k = i
    bader_max = np.zeros((0, 3), dtype=np.int64)
    edge_max = np.zeros((0, 3), dtype=np.int64)
    i_c = np.zeros(1, dtype=np.int64, order='C')
    pbar_tot = np.sum([v[0] for v in vol_s])
    pbar = tqdm_wrap(total=pbar_tot, desc='Bader volume assigning:')
    with ThreadPoolExecutor(max_workers=threads+1) as e:
        f = {
            e.submit(bader, density, vols[i], idx[i], dist_mat, T_grad, i_c): i
            for i in range(thread)
        }
        bar_thread = e.submit(progress_bar_update, pbar, i_c)
        for future in as_completed(f):
            i = f[future]
            vols, max_p, edge_p = future.result()
            volume_offset(vols, bader_max.shape[0], edge_max.shape[0])
            volume_merge(volumes, vols, idx[i], vol_s[i])
            bader_max = array_merge(bader_max, max_p)
            edge_max = array_merge(edge_max, edge_p)
        while i_c[0] < pbar_tot:
            i_c[0] += 1
        if edge_max.shape[0] > 0:
            edge_assign(edge_max, volumes)
    if volumes.dtype != dtype_calc(-bader_max.shape[0]):
        volumes = dtype_change(volumes,
                np.zeros(volumes.shape, dtype=dtype_calc(-bader_max.shape[0])))
    return bader_max, volumes


def assign_to_atoms(bader_max, atoms, lattice, volumes, threads):
    """Assign Bader volumes to atoms.

    args:
        bader_max: cartesian coordinates of Bader maxima
        atoms: the atoms positions in cartesian coordinates
        lattice: the lattice describing the periodic cell
        volumes: voxel to vol_num map
        threads: the number of threads to distribute over (Note: total number
                 will be threads + 1 as progress bar takes a thread)
    returns:
        bader_atoms: array of length Bader volumes linking Bader volume (index)
                     and atom (value)
        bader_distance: array of same length linking Bader volume (index) and
                     distance to atom (value)
        atoms_volumes: voxel to atom map
    """
    thread = max([1, threads])
    atom_volumes = volumes.copy()
    bader_atoms = np.zeros(bader_max.shape[0], dtype=np.int64)
    bader_distance = np.zeros(bader_max.shape[0], dtype=np.float64)
    b_max = np.array_split(bader_max, thread)
    idx = np.zeros(thread, dtype=np.int64)
    for i in range(thread - 1):
        idx[i+1] += idx[i] + b_max[i].shape[0]
    i_c = np.zeros(1, dtype=np.int64, order='C')
    pbar_tot = bader_max.shape[0] + volumes.shape[0]
    pbar = tqdm_wrap(total=pbar_tot, desc='Assigning to atoms:')
    with ThreadPoolExecutor(max_workers=threads + 1) as e:
        f = {
            e.submit(atom_assign, b_max[i], atoms, lattice, i_c): i
            for i in range(thread)
        }
        bar_thread = e.submit(progress_bar_update, pbar, i_c)
        for future in as_completed(f):
            i = f[future]
            atom, dist = future.result()
            bader_atoms[idx[i]:idx[i]+b_max[i].shape[0]] = atom
            bader_distance[idx[i]:idx[i]+b_max[i].shape[0]] = dist
        volume_assign(atom_volumes, bader_atoms, i_c)
        while i_c[0] < pbar_tot:
            i_c[0] += 1
    if atom_volumes.dtype != dtype_calc(-atoms.shape[0]):
        atom_volumes = dtype_change(atom_volumes,
                np.zeros(atom_volumes.shape, dtype=dtype_calc(-atoms.shape[0])))
    return bader_atoms, bader_distance, atom_volumes


def refine(method, refine_mode, density, volumes, dist_mat, T_grad, threads):
    """Set up arrays for threading the Bader edge refinement method.

    args:
        method: name of method (name for name in refinement.__contains__)
        refine_mode: tuple of how to refine the edges (all | changed) and how m
                     any iterations to refine for passing a negative value will
                     refine until no edges change
        density, volumes, dist_mat, T_grad: input for method
        threads: the number of threads to distribute over (Note: total number
                 will be threads + 1 as progress bar takes a thread)
    """
    try:
        refine = getattr(refinement, method)
    except AttributeError:
        return
    check_mode, iters = tuple(refine_mode)
    thread = max([1, threads])
    if iters == 0:
        return
    print(f"\n  Refining {check_mode} edges:")
    known = np.zeros(density.shape, dtype=np.int8)
    edges = refinement.edge_find(known, density, volumes)
    if edges == 0:
        print("  No edges found.")
        return
    zipped = zip(density.shape, factor_3d(thread))
    split = np.array([x for _, x in sorted(zipped, key=lambda x: x[0])])
    knowns = [known]
    for i in range(3):
        rknown = knowns.copy()
        knowns = list()
        for k in rknown:
            kt = np.array_split(k, split[i], axis=i)
            for val in kt:
                knowns.append(np.ascontiguousarray(val))
    rknown = known.copy()
    known_s = np.array([k.shape for k in knowns])
    idx = np.zeros((thread, 3), dtype=dtype_calc(max(density.shape)), order='C')
    c = -1
    k = (0, 0, 0)
    for i in np.ndindex(*split):
        j = np.where(np.subtract(i, k) != 0, 1, 0)
        ii = np.mod(np.add(idx[c], np.multiply(j, known_s[c])), density.shape)
        c += 1
        idx[c] = ii
        k = i
    i_c = np.zeros(1, dtype=np.int64, order='C')
    print("  Iteration 1:")
    pbar_tot = edges
    pbar = tqdm_wrap(total=pbar_tot, desc=f"Refining {edges} edges:")
    changed = 0
    with ThreadPoolExecutor(max_workers=threads+1) as e:
        f = {
            e.submit(refine, knowns[i], rknown, density, volumes, idx[i],
                    dist_mat, T_grad, i_c): i for i in range(thread)
        }
        bar_thread = e.submit(progress_bar_update, pbar, i_c)
        for future in as_completed(f):
            i = f[future]
            knw, chngd = future.result()
            volume_merge(known, knw, idx[i], known_s[i])
            changed += chngd
        while i_c[0] < pbar_tot:
            i_c[0] += 1
    print(f"  {changed} points changed.")
    if iters < 0:
        iters = np.float64('inf')
    iter_num = 2
    while iter_num <= iters:
        print(f"  Iteration {iter_num}:")
        del knw
        del chngd
        if check_mode.lower() == 'all':
            known = np.zeros(density.shape, dtype=np.int8)
            edges = refinement.edge_find(known, density, volumes)
        else:
            checked, edges = refinement.edge_check(known, density, volumes)
        knowns = [known]
        for i in range(3):
            rknown = knowns.copy()
            knowns = list()
            for k in rknown:
                kt = np.array_split(k, split[i], axis=i)
                for val in kt:
                    knowns.append(val)
        rknown = known.copy()
        i_c = np.zeros(1, dtype=np.int64, order='C')
        pbar_tot = edges
        pbar = tqdm_wrap(total=pbar_tot, desc=f"Refining {edges} edges:")
        changed = 0
        with ThreadPoolExecutor(max_workers=threads+1) as e:
            f = {
                e.submit(refine, knowns[i], rknown, density, volumes,
                        idx[i], dist_mat, T_grad, i_c): i for i in range(thread)
            }
            bar_thread = e.submit(progress_bar_update, pbar, i_c)
            for future in as_completed(f):
                i = f[future]
                knw, chngd = future.result()
                volume_merge(known, knw, idx[i], known_s[i])
                changed += chngd
            while i_c[0] < pbar_tot:
                i_c[0] += 1
        print(f"  {changed} points changed.")
        if changed == 0:
            break
        iter_num += 1


def surface_distance(density, volumes, lattice, atoms, threads):
    """Calculate the minimum distance to volume edge for each Bader atom.

    args:
        density: array of charge/spin density required for edge finding
        volumes: voxel to vol_num map
        lattice: the lattice describing the periodic cell
        atoms: the atoms positions in cartesian coordinates
        threads: the number of threads to distribute over (Note: total number
                 will be threads + 1 as progress bar takes a thread)
    returns:
        array of length atoms containing distance (value) for each atom (index)
    """
    thread = max([1, threads])
    print(f"\n  Calculating min. surface disance:")
    known = np.zeros(volumes.shape, dtype=np.int8)
    edges = refinement.edge_find(known, density, volumes)
    if edges == 0:
        print("  No edges found.")
        return
    zipped = zip(density.shape, factor_3d(thread))
    split = np.array([x for _, x in sorted(zipped, key=lambda x: x[0])])
    shape = np.zeros((*split, 3), dtype=np.int64, order='C')
    idx = np.zeros((*split, 3), dtype=dtype_calc(max(density.shape)), order='C')
    k = (0, 0, 0)
    for i in np.ndindex(*split):
        for j in range(3):
            shape[i][j] = volumes.shape[j] // split[j]
            if i[j] < volumes.shape[j] % split[j]:
                shape[i][j] += 1
        j = np.where(np.subtract(i, k) != 0, 1, 0)
        idx[i] = np.mod(np.add(idx[k], np.multiply(j, shape[k])), known.shape)
        k = i
    idx = idx.reshape((thread, 3))
    shape = shape.reshape((thread, 3))
    i_c = np.zeros(1, dtype=np.int64, order='C')
    pbar_tot = edges
    pbar = tqdm_wrap(total=pbar_tot, desc=f"Measuring {edges} edges:")
    surface_distance = np.zeros((thread, atoms.shape[0]), dtype=np.float64)
    with ThreadPoolExecutor(max_workers=threads+1) as e:
        f = {
            e.submit(surface_dist, idx[i], shape[i], known, volumes, lattice,
                    atoms, i_c): i for i in range(thread)
        }
        bar_thread = e.submit(progress_bar_update, pbar, i_c)
        for future in as_completed(f):
            i = f[future]
            surface_distance[i] = future.result()
        while i_c[0] < pbar_tot:
            i_c[0] += 1
    min_surface_dist = np.zeros(atoms.shape[0], dtype=np.float64)
    max_surface = np.zeros(atoms.shape[0], dtype=np.float64)
    max_surface[:] = 'inf'
    for dist in surface_distance:
        dist[dist == 0] = 'inf'
        idx = np.where(max_surface > dist)
        max_surface[idx] = dist[idx]
        min_surface_dist[idx] = dist[idx]
    return min_surface_dist
