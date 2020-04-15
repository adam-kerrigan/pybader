
"""Bader edge refinement methods.

This module is has __contains__ due to @njit hiding the names of the functions.
All functions in this module should have the same arguments as to help be called 
by refine in the thread_handlers module.
"""
import numpy as np
from numba import njit
from .utils import (
        array_assign,
        volume_extend,
)

__contains__ = ['neargrid']

@njit(cache=True, nogil=True)
def neargrid(known, rknown, density, volumes, idx, dist_mat, T_grad, i_c):
    """Parallel implementation of the neargrid Bader maximum searching.

    The neargrid method for Bader maximum searching. Can be passed a known array
    of smaller size than density to reduce the active area for parallelisation
    purposes. This only looks at 'marked' indices and refines their associated
    volume.

    args:
        known: integrer array indicating edge points: -2 indicates edge, -1 is
               point near edge 2 is known point.
        rknown: read-only copy of full known array.
        density: read-only array of reference charge density.
        volumes: array same shape as density to store bader volume indicators.
        idx: the offset of the origin for the chunk (threading).
        dist_mat: rank-3 tensor of distances for moving in index direction 
        T_grad: transform matrix for converting gradient to direct basis
        i_c: index counter for tqdm bar.
    returns:
        known: the updated known array.
        changed: integer counter of how many voxels changed associated volume.
    """
    # get shapes
    known_shape = np.zeros(3, dtype=np.int64)
    kx, ky, kz = known.shape
    # thread stuff
    extend = np.zeros(3, dtype=np.int64)
    positive_len = np.zeros(3, dtype=np.int64)
    new_positive_len = np.zeros(3, dtype=np.int64)
    negative_len = np.zeros(3, dtype=np.int64)
    new_negative_len = np.zeros(3, dtype=np.int64)
    for j in range(3):
        known_shape[j] = known.shape[j]
        extend[j] = known.shape[j]
        positive_len[j] = known.shape[j]
        new_positive_len[j] = known.shape[j]
    # init array length counters
    path_num = 0
    # init arrays for bader maxima, edge crossings and current path
    path = np.zeros((kx, 3), dtype=idx.dtype)
    # init position arrays
    p = np.zeros(3, dtype=np.int64)
    pt = np.zeros(3, dtype=np.int64)
    pd = np.zeros(3, dtype=np.int64)
    pk = np.zeros(3, dtype=np.int64)
    dr = np.zeros(3, dtype=np.float64)
    density_t = np.zeros(2, dtype=np.float64)
    grad = np.zeros(3, dtype=np.float64)
    grad_dir = np.zeros(3, dtype=np.float64)
    max_grad = np.float64(0.)
    # keep track of the lead index for filling the progress bar
    lead_idx = 0
    changed = 0
    # for index in range of the volume size
    for i in np.ndindex(kx, ky, kz):
        # skip if not edge, this should also skip vacuum
        if known[i] != -2:
            continue
        i_c[0] += 1
        # init p for current point, pk for next point in volume space
        # pd for next point in density space
        for j in range(3):
            p[j] = i[j] + idx[j]
            path[0][j] = i[j]
            dr[j] = 0.
            pd[j] = p[j]
        vol_num = volumes[p[0], p[1], p[2]]
        known[i] += 5
        # path size is now 1 and max_val is current point
        path_num = 1
        while True:
            # intialise point in read_only space
            max_val = density[p[0], p[1], p[2]]
            # calculate density of heptacube around point
            for j in range(3):
                # convert to read_only space
                pd[j] += 1
                # wrap in pbc
                if pd[j] < 0:
                    pd[j] += density.shape[j]
                elif pd[j] >= density.shape[j]:
                    pd[j] -= density.shape[j]
                # store density at p+1
                density_t[0] = density[pd[0], pd[1], pd[2]]
                pd[j] -= 2
                # rewrap
                if pd[j] < 0:
                    pd[j] += density.shape[j]
                elif pd[j] >= density.shape[j]:
                    pd[j] -= density.shape[j]
                # store density of p-1
                density_t[1] = density[pd[0], pd[1], pd[2]]
                # if p is max in this axis grad is zero
                # else grad is density[p+1] - density[p-1] / 2
                if density_t[0] < max_val > density_t[1]:
                    grad[j] = 0.
                else:
                    grad[j] = (density_t[0] - density_t[1]) / 2.
                # reset current pd
                pd[j] += 1
                if pd[j] < 0:
                    pd[j] += density.shape[j]
                elif pd[j] >= density.shape[j]:
                    pd[j] -= density.shape[j]
            # convert grad to direct coords
            max_grad = 0.
            for j in range(3):
                grad_dir[j] = ((T_grad[j][0] * grad[0])
                             + (T_grad[j][1] * grad[1])
                             + (T_grad[j][2] * grad[2]))
                if grad_dir[j] > max_grad:
                    max_grad = grad_dir[j]
                elif -grad_dir[j] > max_grad:
                    max_grad = -grad_dir[j]
            # max grad is zero then do ongrid step
            if max_grad < 1E-14:
                for j in range(3):
                    pk[j] = p[j] - idx[j]
            else:
                for j in range(3):
                    grad_dir[j] /= max_grad
                    if grad_dir[j] > 0:
                        int_grad = np.int64(grad_dir[j] + .5)
                    else:
                        int_grad = np.int64(grad_dir[j] - .5)
                    pd[j] = p[j] + int_grad
                    dr[j] += grad_dir[j] - int_grad
                    if dr[j] > 0:
                        int_dr = np.int64(dr[j] + .5)
                    else:
                        int_dr = np.int64(dr[j] - .5)
                    pd[j] += int_dr
                    dr[j] -= int_dr
                    if pd[j] >= density.shape[j]:
                        pd[j] -= density.shape[j]
                    elif pd[j] < 0:
                        pd[j] += density.shape[j]
                    pk[j] = pd[j] - idx[j]
            # check if new pk is same as p or outside of volume space
            extend_flag = False
            for j in range(3):
                # outside volume to left
                if pk[j] < negative_len[j]:
                    upper = pk[j] + density.shape[j] - positive_len[j] + 1
                    lower = (pk[j] - negative_len[j]) * -1
                    if upper <= 0:
                        pk[j] += density.shape[j]
                    elif upper > lower:
                        new_negative_len[j] -= known_shape[j] // 2
                        extend[j] += known_shape[j] // 2
                        extend_flag = True
                    else:
                        new_positive_len[j] += known_shape[j] // 2
                        extend[j] += known_shape[j] // 2
                        pk[j] += density.shape[j]
                        extend_flag = True
                elif pk[j] >= positive_len[j]:
                    upper = pk[j] - positive_len[j] + 1
                    lower = (pk[j] - density.shape[j] - negative_len[j]) * -1
                    if lower <= 0:
                        pk[j] -= density.shape[j]
                    elif upper > lower:
                        new_negative_len[j] -= known_shape[j] // 2
                        extend[j] += known_shape[j] // 2
                        pk[j] -= density.shape[j]
                        extend_flag = True
                    else:
                        new_positive_len[j] += known_shape[j] // 2
                        extend[j] += known_shape[j] // 2
                        extend_flag = True
            if extend_flag:
                for j in range(3):
                    if extend[j] > density.shape[j]:
                        extend[j] = density.shape[j]
                known = volume_extend(known, positive_len, extend)
                for j in range(3):
                    if known.shape[j] == density.shape[j]:
                        positive_len[j] = density.shape[j]
                        negative_len[j] = 0
                    else:
                        positive_len[j] = new_positive_len[j]
                        negative_len[j] = new_negative_len[j]
            # already been here this path
            if 3 <= known[pk[0], pk[1], pk[2]] <= 5:
                for j in range(3):
                    dr[j] = 0.
                    pd[j] = p[j]
                max_val = density[p[0], p[1], p[2]]
                ctr_val = max_val
                for ix in range(-1, 2):
                    # shift p_x into density space and adjust for pbc
                    pt[0] = p[0] + ix
                    if pt[0] < 0:
                        pt[0] += density.shape[0]
                    elif pt[0] >= density.shape[0]:
                        pt[0] -= density.shape[0]
                    for iy in range(-1, 2):
                        # shift p_y into density space and adjust for pbc
                        pt[1] = p[1] + iy
                        if pt[1] < 0:
                            pt[1] += density.shape[1]
                        elif pt[1] >= density.shape[1]:
                            pt[1] -= density.shape[1]
                        for iz in range(-1, 2):
                            # shift p_z into density space and adjust for pbc
                            pt[2] = p[2] + iz
                            if pt[2] < 0:
                                pt[2] += density.shape[2]
                            elif pt[2] >= density.shape[2]:
                                pt[2] -= density.shape[2]
                            # check for new maxima, save density and index
                            pd_val = density[pt[0], pt[1], pt[2]]
                            pd_val = (pd_val - ctr_val) * dist_mat[ix, iy, iz]
                            pd_val += ctr_val
                            if pd_val > max_val:
                                max_val = pd_val
                                for j in range(3):
                                    pd[j] = pt[j]
                                    pk[j] = pd[j] - idx[j]
                extend_flag = False
                break_flag = True
                for j in range(3):
                    # outside volume to left
                    if pk[j] < negative_len[j]:
                        upper = pk[j] + density.shape[j] - positive_len[j] + 1
                        lower = (pk[j] - negative_len[j]) * -1
                        if upper <= 0:
                            pk[j] += density.shape[j]
                        elif upper > lower:
                            new_negative_len[j] -= known_shape[j] // 2
                            extend[j] += known_shape[j] // 2
                            extend_flag = True
                        else:
                            new_positive_len[j] += known_shape[j] // 2
                            extend[j] += known_shape[j] // 2
                            pk[j] += density.shape[j]
                            extend_flag = True
                    elif pk[j] >= positive_len[j]:
                        upper = pk[j] - positive_len[j] + 1
                        lower = (pk[j] - density.shape[j] - negative_len[j])
                        lower *= -1
                        if lower <= 0:
                            pk[j] -= density.shape[j]
                        elif upper > lower:
                            new_negative_len[j] -= known_shape[j] // 2
                            extend[j] += known_shape[j] // 2
                            pk[j] -= density.shape[j]
                            extend_flag = True
                        else:
                            new_positive_len[j] += known_shape[j] // 2
                            extend[j] += known_shape[j] // 2
                            extend_flag = True
                    if break_flag and pd[j] != p[j]:
                        break_flag = False
                if extend_flag:
                    for j in range(3):
                        if extend[j] > density.shape[j]:
                            extend[j] = density.shape[j]
                    known = volume_extend(known, positive_len, extend)
                    for j in range(3):
                        if known.shape[j] == density.shape[j]:
                            positive_len[j] = density.shape[j]
                            negative_len[j] = 0 # 1 - density.shape[j]
                        else:
                            positive_len[j] = new_positive_len[j]
                            negative_len[j] = new_negative_len[j]
                if break_flag:
                    for j in range(3):
                        p[j] = i[j] + idx[j]
                    new_vol_num = volumes[pd[0], pd[1], pd[2]]
                    if new_vol_num != vol_num:
                        volumes[p[0], p[1], p[2]] += new_vol_num - vol_num
                        changed += 1
                    else:
                        known[i[0], i[1], i[2]] += 1
                    break
            # if known break without updating p
            if rknown[pd[0], pd[1], pd[2]] == 2:
                for j in range(3):
                    p[j] = i[j] + idx[j]
                new_vol_num = volumes[pd[0], pd[1], pd[2]]
                if new_vol_num != vol_num:
                    volumes[p[0], p[1], p[2]] += new_vol_num - vol_num
                    changed += 1
                else:
                    known[i[0], i[1], i[2]] += 1
                break
            # no break condition so add point to path
            else:
                if path_num >= path.shape[0]:
                    path = array_assign(path, path.shape[0], path.shape[0] + kx)
                for j in range(3):
                    p[j] = pd[j]
                    path[path_num][j] = pk[j]
                path_num += 1
                # we've visited the point
                if known[pk[0], pk[1], pk[2]] < 2:
                    known[pk[0], pk[1], pk[2]] += 5
        # assign bader_num to volumes
        for j in range(path_num):
            for k in range(3):
                pk[k] = path[j][k]
            if known[pk[0], pk[1], pk[2]] > 2:
                known[pk[0], pk[1], pk[2]] += -5
    return known, changed


@njit(cache=True, nogil=True)
def edge_find(known, density, volumes):
    """Searches a volumes array for edges between volumes, ignores vacuum.

    args:
        known: the known array to read and output to
        density: used to check edge isn't also a maxima
        volumes: the volumes array to check for edges in
    returns:
        number of changed edges
    """
    rx, ry, rz = density.shape
    p = np.zeros(3, dtype=np.int64)
    edge_num = 0
    for i in np.ndindex(density.shape):
        if known[i] == 2:
            continue
        if volumes[i] == -1:
            continue
        # set flags and current values for edge detection
        is_max = True
        is_edge = False
        vol_num = volumes[i]
        max_val = density[i]
        for ix in range(-1, 2):
            p[0] = i[0] + ix
            if p[0] < 0:
                p[0] += density.shape[0]
            elif p[0] >= density.shape[0]:
                p[0] -= density.shape[0]
            for iy in range(-1, 2):
                p[1] = i[1] + iy
                if p[1] < 0:
                    p[1] += density.shape[1]
                elif p[1] >= density.shape[1]:
                    p[1] -= density.shape[1]
                for iz in range(-1, 2):
                    p[2] = i[2] + iz
                    if p[2] < 0:
                        p[2] += density.shape[2]
                    elif p[2] >= density.shape[2]:
                        p[2] -= density.shape[2]
                    new_vol_num = volumes[p[0], p[1], p[2]]
                    new_max_val = density[p[0], p[1], p[2]]
                    # should we ignore vacuum here?
                    if new_vol_num == -1:
                        continue
                    if new_vol_num != vol_num:
                        is_edge = True
                    if new_max_val > max_val:
                        is_max = False
        if not is_edge:
            if known[i] >= 0:
                known[i] = 2
        elif is_max:
            if known[i] >= 0:
                known[i] = 2
        else:
            known[i] = -2
            edge_num += 1
            for ix in range(-1, 2):
                p[0] = i[0] + ix
                if p[0] < 0:
                    p[0] += density.shape[0]
                elif p[0] >= density.shape[0]:
                    p[0] -= density.shape[0]
                for iy in range(-1, 2):
                    p[1] = i[1] + iy
                    if p[1] < 0:
                        p[1] += density.shape[1]
                    elif p[1] >= density.shape[1]:
                        p[1] -= density.shape[1]
                    for iz in range(-1, 2):
                        p[2] = i[2] + iz
                        if p[2] < 0:
                            p[2] += density.shape[2]
                        elif p[2] >= density.shape[2]:
                            p[2] -= density.shape[2]
                        if known[p[0], p[1], p[2]] >= 0:
                            known[p[0], p[1], p[2]] = -1
    return edge_num


@njit(cache=True, nogil=True)
def edge_check(known, density, volumes):
    """Searches an already edge flaged volumes array for edges between volumes.

    args:
        known: the known array to read and output to
        density: used to check edge isn't also a maxima
        volumes: the volumes array to check for edges in
    returns:
        (number of checked edges, number of changed edges)
    """
    rx, ry, rz = density.shape
    p = np.zeros(3, dtype=np.int64)
    pe = np.zeros(3, dtype=np.int64)
    checked = 0
    edge_num = 0
    for i in np.ndindex(known.shape):
        # if known or vacuum skip
        if known[i] != -2:
            continue
        for ex in range(-1, 2):
            pe[0] = i[0] + ex
            if pe[0] < 0:
                pe[0] += density.shape[0]
            elif pe[0] >= density.shape[0]:
                pe[0] -= density.shape[0]
            for ey in range(-1, 2):
                pe[1] = i[1] + ey
                if pe[1] < 0:
                    pe[1] += density.shape[1]
                elif pe[1] >= density.shape[1]:
                    pe[1] -= density.shape[1]
                for ez in range(-1, 2):
                    pe[2] = i[2] + ez
                    if pe[2] < 0:
                        pe[2] += density.shape[2]
                    elif pe[2] >= density.shape[2]:
                        pe[2] -= density.shape[2]
                    is_max = True
                    is_edge = False
                    vol_num = volumes[pe[0], pe[1], pe[2]]
                    max_val = density[pe[0], pe[1], pe[2]]
                    for ix in range(-1, 2):
                        p[0] = pe[0] + ix
                        if p[0] < 0:
                            p[0] += density.shape[0]
                        elif p[0] >= density.shape[0]:
                            p[0] -= density.shape[0]
                        for iy in range(-1, 2):
                            p[1] = pe[1] + iy
                            if p[1] < 0:
                                p[1] += density.shape[1]
                            elif p[1] >= density.shape[1]:
                                p[1] -= density.shape[1]
                            for iz in range(-1, 2):
                                p[2] = pe[2] + iz
                                if p[2] < 0:
                                    p[2] += density.shape[2]
                                elif p[2] >= density.shape[2]:
                                    p[2] -= density.shape[2]
                                new_vol_num = volumes[p[0], p[1], p[2]]
                                new_max_val = density[p[0], p[1], p[2]]
                                # should we ignore vacuum here?
                                if new_vol_num == -1:
                                    continue
                                if new_vol_num != vol_num:
                                    is_edge = True
                                if new_max_val > max_val:
                                    is_max = False
                    if not is_edge:
                        known[pe[0], pe[1], pe[2]] = -1
                        checked += 1
                    elif not is_max:
                        if known[pe[0], pe[1], pe[2]] != -3:
                            known[pe[0], pe[1], pe[2]] = -3
                            edge_num += 1
                            for ix in range(-1, 2):
                                p[0] = pe[0] + ix
                                if p[0] < 0:
                                    p[0] += density.shape[0]
                                elif p[0] >= density.shape[0]:
                                    p[0] -= density.shape[0]
                                for iy in range(-1, 2):
                                    p[1] = pe[1] + iy
                                    if p[1] < 0:
                                        p[1] += density.shape[1]
                                    elif p[1] >= density.shape[1]:
                                        p[1] -= density.shape[1]
                                    for iz in range(-1, 2):
                                        p[2] = pe[2] + iz
                                        if p[2] < 0:
                                            p[2] += density.shape[2]
                                        elif p[2] >= density.shape[2]:
                                            p[2] -= density.shape[2]
                                        if known[p[0], p[1], p[2]] >= 0:
                                            known[p[0], p[1], p[2]] = -1
                            checked += 1
    for i in np.ndindex(known.shape):
        if known[i] == -3:
            known[i] += 1
    return checked, edge_num
