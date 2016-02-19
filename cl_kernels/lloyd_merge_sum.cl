/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 * 
 * 
 * Copyright (c) 2016, Lutz, Clemens <lutzcle@cml.li>
 */

#ifdef TYPE32
#define CL_FP float
#define CL_INT uint
#else
#ifdef TYPE64
#define CL_FP double
#define CL_INT ulong
#endif
#endif

CL_INT ccoord2ind(CL_INT dim, CL_INT row, CL_INT col) {
    return dim * col + row;
}

CL_INT rcoord2ind(CL_INT dim, CL_INT row, CL_INT col) {
    return dim * row + col;
}

CL_INT div_round_up(CL_INT dividend, CL_INT divisor) {
    return (dividend + divisor - 1) / divisor;
}

/*
 * Update centroids
 * 
 * g_centroids: global_size
 * l_points: local_size
 */
__kernel
void lloyd_merge_sum(
        __global CL_FP const *const restrict g_points,
        __global CL_FP *const restrict g_centroids,
        __global CL_INT const *const restrict g_mass,
        __global CL_INT const *const restrict g_labels,
        __local CL_FP *const restrict l_points,
        __local CL_INT *const restrict l_mass,
        __local CL_INT *const restrict l_labels,
        const CL_INT NUM_FEATURES,
        const CL_INT NUM_POINTS,
        const CL_INT NUM_CLUSTERS
       ) {

    CL_INT const num_local_cols = max((CL_INT)1, (CL_INT)get_local_size(0) / NUM_CLUSTERS);
    CL_INT const num_local_rows_points = get_local_size(0) / num_local_cols;
    CL_INT const num_local_rows_clusters = min(NUM_CLUSTERS, num_local_rows_points);

    CL_INT const num_global_blocks = get_global_size(0) / (num_local_rows_points * num_local_cols);
    CL_INT const g_block = (get_group_id(0) * get_local_size(0)) / (num_local_rows_points * num_local_cols);

    CL_INT const num_tile_cols = div_round_up(NUM_FEATURES, num_local_cols);
    CL_INT const num_tile_rows = div_round_up(NUM_CLUSTERS, num_local_rows_clusters);

    CL_INT const l_col = get_local_id(0) / num_local_rows_points;
    CL_INT const l_row = get_local_id(0) % num_local_rows_points;

    CL_INT const t_col = (get_global_id(0) / (num_local_cols * num_local_rows_clusters * num_tile_rows)) % num_tile_cols;
    CL_INT const t_row = (get_global_id(0) / num_local_rows_clusters) % num_tile_rows;

    CL_INT const b_col = l_col + num_local_cols * t_col;
    CL_INT const b_row = l_row + num_local_rows_clusters * t_row;

    if (b_col >= NUM_FEATURES) {
        return;
    }

    CL_FP centroid = 0;

    for (CL_INT p = num_local_rows_points * g_block; p < NUM_POINTS; p += num_local_rows_points * num_global_blocks) {
        // p: Current point global index

        if (p + l_row < NUM_POINTS) {
            if (l_col == 0) {
                l_labels[l_row] = g_labels[p + l_row];
            }

            l_points[ccoord2ind(num_local_rows_points, l_row, l_col)] = g_points[ccoord2ind(NUM_POINTS, p + l_row, b_col)];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        if (b_row < NUM_CLUSTERS && l_row < num_local_rows_clusters) {
            for (CL_INT r = 0; r < num_local_rows_points && p + r < NUM_POINTS; ++r) {
                if (l_labels[r] == b_row) {
                    centroid += l_points[ccoord2ind(num_local_rows_points, r, l_col)];
                }
            }
        }
    }

    if (b_row < NUM_CLUSTERS && l_row < num_local_rows_clusters) {
        if (l_col == 0) {
            l_mass[l_row] = g_mass[b_row];
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (b_row < NUM_CLUSTERS && l_row < num_local_rows_clusters) {
        CL_INT mass = l_mass[l_row];
        CL_INT tile_ind = ccoord2ind(NUM_CLUSTERS, b_row, b_col);
        CL_INT global_ind = tile_ind + NUM_CLUSTERS * NUM_FEATURES * g_block;
        g_centroids[global_ind] = centroid / mass;
    }
}
