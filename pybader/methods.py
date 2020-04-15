"""Bader calculation methods.

This module is has __contains__ due to @njit hiding the names of the functions.
All functions in this module should have the same arguments as to help be called
by bader_calc in the thread_handlers module.
"""
import numpy as np
from numba import njit
from .utils import (
        array_assign,
        volume_extend,
)

__contains__ = ['ongrid', 'neargrid']


@njit(cache=True, nogil=True)
def ongrid(density, volumes, idx, dist_mat, T_grad, i_c):
    """Parallel implementation of the ongrid Bader maximum searching.

    The ongrid method for Bader maximum searching based on W. Tang, E. Sanville,
    and G. Henkelman A grid-based Bader analysis algorithm without lattice bias,
    J. Phys.: Condens. Matter 21, 084204 (2009). Can be passed a volumes array
    of smaller size than density to reduce the active area for parallelisation
    purposes.

    args:
        density: read-only array of reference charge density
        volumes: array same shape as density to store bader volume indicators
        idx: the offset of the origin for the chunk (threading)
        dist_mat: rank-3 tensor of distances for moving in index direction
        T_grad: unused, kept for argument matching with neargrid method
        i_c: index counter for tqdm bar.
    returns:
        volumes: the updated volumes array
        bader_max: array containing location of Bader maxima
        edge_max: array containing location of edge crossings
    """
    # get shapes
    vol_shape = np.zeros(3, dtype=np.int64)
    vx, vy, vz = volumes.shape
    # thread stuff
    extend = np.zeros(3, dtype=np.int64)
    positive_len = np.zeros(3, dtype=np.int64)
    new_positive_len = np.zeros(3, dtype=np.int64)
    negative_len = np.zeros(3, dtype=np.int64)
    new_negative_len = np.zeros(3, dtype=np.int64)
    for j in range(3):
        vol_shape[j] = volumes.shape[j]
        extend[j] = volumes.shape[j]
        positive_len[j] = volumes.shape[j]
        new_positive_len[j] = volumes.shape[j]
    # init array length counters
    bader_num = 0
    edge_num = 0
    path_num = 0
    # init arrays for bader maxima, edge crossings and current path
    # idx is type set for the maximum int that can be stored here
    bader_max = np.zeros((vx, 3), dtype=idx.dtype)
    edge_max = np.zeros((vx, 3), dtype=idx.dtype)
    path = np.zeros((vx, 3), dtype=idx.dtype)
    # init position arrays
    p = np.zeros(3, dtype=np.int64)
    pt = np.zeros(3, dtype=np.int64)
    pd = np.zeros(3, dtype=np.int64)
    pv = np.zeros(3, dtype=np.int64)
    # keep track of the lead index for filling the progress bar
    lead_idx = 0
    # for index in range of the volume size
    for i in np.ndindex(vx, vy, vz):
        if i[0] != lead_idx:
            lead_idx += 1
            i_c[0] += 1
        # skip if volume has been visited
        if volumes[i] != 0:
            continue
        # init p for current point, pv for next point in volume space
        # pd for next point in density space
        for j in range(3):
            p[j] = i[j] + idx[j]
            pv[j] = i[j]
            pd[j] = p[j]
            path[0][j] = pv[j]
        # path size is now 1 and max_val is current point
        path_num = 1
        max_val = density[p[0], p[1], p[2]]
        ctr_val = max_val
        while True:
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
                        pd_tmp = density[pt[0], pt[1], pt[2]]
                        pd_val = (pd_tmp - ctr_val) * dist_mat[ix, iy, iz]
                        pd_val += ctr_val
                        if pd_val > max_val:
                            max_val = pd_val
                            new_density = pd_tmp
                            for j in range(3):
                                pd[j] = pt[j]
                                pv[j] = pd[j] - idx[j]
            # check if new pv is same as p or outside of volume space
            break_flag = True
            extend_flag = False
            for j in range(3):
                # outside volume to left
                if pv[j] < negative_len[j]:
                    upper = pv[j] + density.shape[j] - positive_len[j] + 1
                    lower = (pv[j] - negative_len[j]) * -1
                    if upper <= 0:
                        pv[j] += density.shape[j]
                    elif upper > lower:
                        new_negative_len[j] -= vol_shape[j] // 2
                        extend[j] += vol_shape[j] // 2
                        extend_flag = True
                    else:
                        new_positive_len[j] += vol_shape[j] // 2
                        extend[j] += vol_shape[j] // 2
                        pv[j] += density.shape[j]
                        extend_flag = True
                elif pv[j] >= positive_len[j]:
                    upper = pv[j] - positive_len[j] + 1
                    lower = (pv[j] - density.shape[j] - negative_len[j]) * -1
                    if lower <= 0:
                        pv[j] -= density.shape[j]
                    elif upper > lower:
                        new_negative_len[j] -= vol_shape[j] // 2
                        extend[j] += vol_shape[j] // 2
                        pv[j] -= density.shape[j]
                        extend_flag = True
                    else:
                        new_positive_len[j] += vol_shape[j] // 2
                        extend[j] += vol_shape[j] // 2
                        extend_flag = True
                if break_flag and pd[j] != p[j]:
                    break_flag = False
            if extend_flag:
                for j in range(3):
                    if extend[j] > density.shape[j]:
                        extend[j] = density.shape[j]
                volumes = volume_extend(volumes, positive_len, extend)
                for j in range(3):
                    if volumes.shape[j] == density.shape[j]:
                        positive_len[j] = density.shape[j]
                        negative_len[j] = 0
                    else:
                        positive_len[j] = new_positive_len[j]
                        negative_len[j] = new_negative_len[j]
            # if known break without updating p
            if volumes[pv[0], pv[1], pv[2]] != 0:
                vol_num = volumes[pv[0], pv[1], pv[2]]
                break
            elif break_flag:
                # store maxima/edge in density space
                vol_num = 0
                for k in range(3):
                    if pv[k] >= vol_shape[k]:
                        vol_num = -2
                    elif pv[k] < 0:
                        vol_num = -2
                break
            # no break condition so add point to path
            else:
                if path_num >= path.shape[0]:
                    path = array_assign(path, path.shape[0], path.shape[0] + vx)
                for j in range(3):
                    p[j] = pd[j]
                    path[path_num][j] = pv[j]
                path_num += 1
                # set new density values for max and control
                ctr_val = new_density
                max_val = new_density
        # if the volume is empty then create a new one
        if vol_num == -2:
            if edge_num >= edge_max.shape[0]:
                edge_max = array_assign(edge_max, edge_max.shape[0],
                        edge_max.shape[0] + vx)
            # add max to bader_max list add one to len counter
            for j in range(3):
                edge_max[edge_num][j] = pd[j]
            edge_num += 1
            vol_num = -2 - edge_num  # -1 is vacuum, -2 is maxima flag
        # we are at a maxima
        elif vol_num == 0:
            if bader_num >= bader_max.shape[0]:
                bader_max = array_assign(bader_max, bader_max.shape[0],
                        bader_max.shape[0] + vx)
            # add max to bader_max list add one to len counter
            for j in range(3):
                bader_max[bader_num][j] = pd[j]
            bader_num += 1
            vol_num = bader_num
        # assign bader_num to volumes
        for j in range(path_num):
            for k in range(3):
                pv[k] = path[j][k]
            volumes[pv[0], pv[1], pv[2]] = vol_num
    # reduce size of bader_max and edge_max arrays to fit contents
    bader_max = array_assign(bader_max, bader_num, bader_num)
    edge_max = array_assign(edge_max, edge_num, edge_num)
    i_c[0] += 1
    return volumes, bader_max, edge_max


@njit(cache=True, nogil=True)
def neargrid(density, volumes, idx, dist_mat, T_grad, i_c):
    """Parallel implementation of the neargrid Bader maximum searching.

    The neargrid method for Bader maximum searching based on W. Tang,
    E. Sanville, and G. Henkelman A grid-based Bader analysis algorithm without
    lattice bias, J. Phys.: Condens. Matter 21, 084204 (2009). Can be passed a
    volumes array of smaller size than density to reduce the active area for
    parallelisation purposes.

    args:
        density: read-only array of reference charge density
        volumes: array same shape as density to store bader volume indicators
        idx: the offset of the origin for the chunk (threading)
        dist_mat: rank-3 tensor of distances for moving in index direction
        T_grad: transform matrix for converting gradient to direct basis
        i_c: index counter for tqdm bar
    returns:
        volumes: the updated volumes array.
        bader_max: array containing location of Bader maxima.
        edge_max: array containing location of edge crossings.
    """
    # get shapes
    vol_shape = np.zeros(3, dtype=np.int64)
    vx, vy, vz = volumes.shape
    # thread stuff
    extend = np.zeros(3, dtype=np.int64)
    positive_len = np.zeros(3, dtype=np.int64)
    new_positive_len = np.zeros(3, dtype=np.int64)
    negative_len = np.zeros(3, dtype=np.int64)
    new_negative_len = np.zeros(3, dtype=np.int64)
    for j in range(3):
        vol_shape[j] = volumes.shape[j]
        extend[j] = volumes.shape[j]
        positive_len[j] = volumes.shape[j]
        new_positive_len[j] = volumes.shape[j]
    # init array length counters
    bader_num = 0
    edge_num = 0
    path_num = 0
    # init arrays for bader maxima, edge crossings and current path
    # idx is type set for the maximum int that can be stored here
    bader_max = np.zeros((vx, 3), dtype=idx.dtype)
    edge_max = np.zeros((vx, 3), dtype=idx.dtype)
    path = np.zeros((vx, 3), dtype=idx.dtype)
    # init position arrays
    p = np.zeros(3, dtype=np.int64)
    pd = np.zeros(3, dtype=np.int64)
    pv = np.zeros(3, dtype=np.int64)
    pt = np.zeros(3, dtype=np.int64)
    dr = np.zeros(3, dtype=np.float64)
    density_t = np.zeros(2, dtype=np.float64)
    grad = np.zeros(3, dtype=np.float64)
    grad_dir = np.zeros(3, dtype=np.float64)
    max_grad = np.float64(0.)
    known = np.zeros((vx, vy, vz), dtype=np.int8)
    # keep track of the lead index for filling the progress bar
    lead_idx = 0
    # for index in range of the volume size
    for i in np.ndindex(vx, vy, vz):
        if i[0] != lead_idx:
            lead_idx += 1
            i_c[0] += 1
        # skip if volume has been visited
        if volumes[i] == -1:
            continue
        elif known[i] == 2:
            continue
        # we've visited the point
        known[i] = 1
        # init p for current point, pv for next point in volume space
        # pd for next point in density space
        for j in range(3):
            p[j] = i[j] + idx[j]
            pd[j] = p[j]
            path[0][j] = i[j]
            dr[j] = 0.
        # path size is now 1 and max_val is current point
        path_num = 1
        while True:
            max_val = density[p[0], p[1], p[2]]
            # calculate density of heptacube around point
            for j in range(3):
                # convert to density space
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
                if density_t[0] <= max_val >= density_t[1]:
                    grad[j] = 0.
                else:
                    grad[j] = (density_t[0] - density_t[1]) / 2.
                # reset current pd
                pd[j] = p[j]
            # convert grad to direct coords
            max_grad = 0.
            for j in range(3):
                grad_dir[j] = ((T_grad[j, 0] * grad[0])
                             + (T_grad[j, 1] * grad[1])
                             + (T_grad[j, 2] * grad[2]))
                if grad_dir[j] > max_grad:
                    max_grad = grad_dir[j]
                elif -grad_dir[j] > max_grad:
                    max_grad = -grad_dir[j]
            # max grad is zero then do ongrid step
            if max_grad < 1E-14:
                for j in range(3):
                    pv[j] = pd[j] - idx[j]
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
                    pv[j] = pd[j] - idx[j]
            # check if pv is outside of volume space and either extend volume
            # space or wrap back in
            extend_flag = False
            for j in range(3):
                # outside volume to left
                if pv[j] < negative_len[j]:
                    upper = pv[j] + density.shape[j] - positive_len[j] + 1
                    lower = (pv[j] - negative_len[j]) * -1
                    if upper <= 0:
                        pv[j] += density.shape[j]
                    elif upper > lower:
                        new_negative_len[j] -= vol_shape[j] // 2
                        extend[j] += vol_shape[j] // 2
                        extend_flag = True
                    else:
                        new_positive_len[j] += vol_shape[j] // 2
                        extend[j] += vol_shape[j] // 2
                        pv[j] += density.shape[j]
                        extend_flag = True
                elif pv[j] >= positive_len[j]:
                    upper = pv[j] - positive_len[j] + 1
                    lower = (pv[j] - density.shape[j] - negative_len[j]) * -1
                    if lower <= 0:
                        pv[j] -= density.shape[j]
                    elif upper > lower:
                        new_negative_len[j] -= vol_shape[j] // 2
                        extend[j] += vol_shape[j] // 2
                        pv[j] -= density.shape[j]
                        extend_flag = True
                    else:
                        new_positive_len[j] += vol_shape[j] // 2
                        extend[j] += vol_shape[j] // 2
                        extend_flag = True
            if extend_flag:
                for j in range(3):
                    if extend[j] > density.shape[j]:
                        extend[j] = density.shape[j]
                volumes = volume_extend(volumes, positive_len, extend)
                known = volume_extend(known, positive_len, extend)
                for j in range(3):
                    if volumes.shape[j] == density.shape[j]:
                        positive_len[j] = density.shape[j]
                        negative_len[j] = 0
                    else:
                        positive_len[j] = new_positive_len[j]
                        negative_len[j] = new_negative_len[j]
            # already been here this path
            if known[pv[0], pv[1], pv[2]] == 1:
                for j in range(3):
                    dr[j] = 0.
                    pd[j] = p[j]
                    pv[j] = p[j] - idx[j]
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
                                    pv[j] = pd[j] - idx[j]
                extend_flag = False
                break_flag = True
                for j in range(3):
                    # outside volume to left
                    if pv[j] < negative_len[j]:
                        upper = pv[j] + density.shape[j] - positive_len[j] + 1
                        lower = (pv[j] - negative_len[j]) * -1
                        if upper <= 0:
                            pv[j] += density.shape[j]
                        elif upper > lower:
                            new_negative_len[j] -= vol_shape[j] // 2
                            extend[j] += vol_shape[j] // 2
                            extend_flag = True
                        else:
                            new_positive_len[j] += vol_shape[j] // 2
                            extend[j] += vol_shape[j] // 2
                            pv[j] += density.shape[j]
                            extend_flag = True
                    elif pv[j] >= positive_len[j]:
                        upper = pv[j] - positive_len[j] + 1
                        lower = (pv[j] - density.shape[j] - negative_len[j])
                        lower *= -1
                        if lower <= 0:
                            pv[j] -= density.shape[j]
                        elif upper > lower:
                            new_negative_len[j] -= vol_shape[j] // 2
                            extend[j] += vol_shape[j] // 2
                            pv[j] -= density.shape[j]
                            extend_flag = True
                        else:
                            new_positive_len[j] += vol_shape[j] // 2
                            extend[j] += vol_shape[j] // 2
                            extend_flag = True
                    if break_flag and pd[j] != p[j]:
                        break_flag = False
                if extend_flag:
                    for j in range(3):
                        if extend[j] > density.shape[j]:
                            extend[j] = density.shape[j]
                    volumes = volume_extend(volumes, positive_len, extend)
                    known = volume_extend(known, positive_len, extend)
                    for j in range(3):
                        if volumes.shape[j] == density.shape[j]:
                            positive_len[j] = density.shape[j]
                            negative_len[j] = 0
                        else:
                            positive_len[j] = new_positive_len[j]
                            negative_len[j] = new_negative_len[j]
                if break_flag:
                    # store maxima/edge in density space
                    vol_num = 0
                    if volumes[pv[0], pv[1], pv[2]] != 0:
                        vol_num = volumes[pv[0], pv[1], pv[2]]
                    else:
                        for k in range(3):
                            if pv[k] >= vol_shape[k]:
                                vol_num = -2
                            elif pv[k] < 0:
                                vol_num = -2
                    break
            # if known break without updating p
            if known[pv[0], pv[1], pv[2]] == 2:
                vol_num = volumes[pv[0], pv[1], pv[2]]
                break
            # no break condition so add point to path
            else:
                if path_num >= path.shape[0]:
                    path = array_assign(path, path.shape[0], path.shape[0] + vx)
                for j in range(3):
                    p[j] = pd[j]
                    path[path_num][j] = pv[j]
                path_num += 1
                known[pv[0], pv[1], pv[2]] = 1
        # if the volume is empty then create a new one
        if vol_num == -2:
            if edge_num >= edge_max.shape[0]:
                edge_max = array_assign(edge_max, edge_max.shape[0],
                        edge_max.shape[0] + vx)
            # add max to bader_max list add one to len counter
            for j in range(3):
                edge_max[edge_num][j] = pd[j]
            edge_num += 1
            vol_num = -2 - edge_num  # -1 is vacuum, -2 is maxima flag
        # we are at a maxima
        elif vol_num == 0:
            if bader_num >= bader_max.shape[0]:
                bader_max = array_assign(bader_max, bader_max.shape[0],
                        bader_max.shape[0] + vx)
            # add max to bader_max list add one to len counter
            for j in range(3):
                bader_max[bader_num][j] = pd[j]
            bader_num += 1
            vol_num = bader_num
        # assign bader_num to volumes and adjust known
        for j in range(path_num):
            for k in range(3):
                p[k] = path[j][k]
                pv[k] = p[k]
                pt[k] = p[k]
            volumes[p[0], p[1], p[2]] = vol_num
            # this should never == 2 ?
            if known[p[0], p[1], p[2]] != 2:
                known[p[0], p[1], p[2]] = 0
            for k in range(3):
                pv[k] += 1
                pt[k] += 1
                # pv[k] check is in bounds, if not we havent been there so skip
                if negative_len[k] <= pv[k] < positive_len[k]:
                    known_flag = True
                    vol_temp = volumes[pv[0], pv[1], pv[2]]
                    if not (-2 < vol_temp < 1):
                        for h in range(3):
                            pt[h] += 1
                            if not (negative_len[h] <= pt[h] < positive_len[h]):
                                known_flag = False
                                break
                            elif vol_temp != volumes[pt[0], pt[1], pt[2]]:
                                known_flag = False
                                break
                            pt[h] -= 2
                            if not (negative_len[h] <= pt[h] < positive_len[h]):
                                known_flag = False
                                break
                            elif vol_temp != volumes[pt[0], pt[1], pt[2]]:
                                known_flag = False
                                break
                            pt[h] += 1
                        if known_flag:
                            known[pv[0], pv[1], pv[2]] = 2
                pv[k] -= 2
                for h in range(3):
                    pt[h] = pv[h]
                if negative_len[k] <= pv[k] < positive_len[k]:
                    # pv[k] check in bounds, if not we havent been there so skip
                    known_flag = True
                    vol_temp = volumes[pv[0], pv[1], pv[2]]
                    if not (-2 < vol_temp < 1):
                        for h in range(3):
                            pt[h] += 1
                            if not (negative_len[h] <= pt[h] < positive_len[h]):
                                known_flag = False
                                break
                            elif vol_temp != volumes[pt[0], pt[1], pt[2]]:
                                known_flag = False
                                break
                            pt[h] -= 2
                            if not (negative_len[h] <= pt[h] < positive_len[h]):
                                known_flag = False
                                break
                            elif vol_temp != volumes[pt[0], pt[1], pt[2]]:
                                known_flag = False
                                break
                            pt[h] += 1
                        if known_flag:
                            known[pv[0], pv[1], pv[2]] = 2
                pv[k] += 1
                for h in range(3):
                    pt[h] = pv[h]
    # reduce size of bader_max and edge_max arrays to fit contents
    bader_max = array_assign(bader_max, bader_num, bader_num)
    edge_max = array_assign(edge_max, edge_num, edge_num)
    i_c[0] += 1
    return volumes, bader_max, edge_max
