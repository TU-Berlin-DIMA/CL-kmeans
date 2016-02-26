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
void lloyd_merge_blocks(
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

    CL_INT const centroids_size = NUM_FEATURES * NUM_POINTS;
    CL_INT const num_local_blocks = get_local_size(0) / centroids_size;
    CL_INT const num_global_blocks = get_global_size(0) / centroids_size;
    CL_INT const l_block = get_local_id(0) / centroids_size;
    CL_INT const g_block = get_global_id(0) / centroids_size;
    CL_INT const l_pos = get_local_id(0) - l_block * centroids_size;
    CL_INT const l_cluster = l_pos % NUM_CLUSTERS;
    CL_INT const l_feature = l_pos / NUM_CLUSTERS;
    CL_INT const l_point = get_local_id(0) % (get_local_size(0) / NUM_FEATURES);

    if (b_col >= NUM_FEATURES) {
        return;
    }

    CL_FP centroid = 0;

    for (CL_INT r = 0; r < NUM_POINTS; r += get_global_size(0)) {
        // p: Current point global index
        CL_INT p = r + get_global_id(0);

        if (p < NUM_POINTS) {
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

__kernel
void lloyd_merge_tiles() {
}
