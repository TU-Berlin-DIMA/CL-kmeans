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

__kernel
void lloyd_feature_sum_pardim(
        __global CL_FP const *const restrict g_points,
        __global CL_FP *const restrict g_centroids,
        __global CL_INT const *const restrict g_mass,
        __global CL_INT const *const restrict g_labels,
        __local CL_FP *const restrict l_centroids,
        CL_INT const NUM_FEATURES,
        CL_INT const NUM_POINTS,
        CL_INT const NUM_CLUSTERS
        ) {

    CL_INT block = get_local_id(0);
    CL_INT block_offset = block * get_local_size(1)
        * NUM_THREAD_FEATURES * NUM_CLUSTERS;
    CL_INT tile_row = get_global_id(0);
    CL_INT tile_col = get_global_id(1);
    CL_INT num_col_tiles = get_global_size(1);
    CL_INT l_feature_base = NUM_THREAD_FEATURES * get_local_id(1);
    CL_INT g_feature_base = NUM_THREAD_FEATURES * get_global_id(1);
    CL_INT g_cluster_offset = (num_col_tiles * tile_row)
        * NUM_THREAD_FEATURES * NUM_CLUSTERS;

    for (CL_INT f = 0; f < NUM_THREAD_FEATURES; ++f) {
        for (CL_INT c = 0; c < NUM_CLUSTERS; ++c) {
            l_centroids[
                block_offset + ccoord2ind(NUM_CLUSTERS, c, l_feature_base + f)
            ] = 0;
        }
    }

    for (CL_INT f = 0; f < NUM_THREAD_FEATURES; ++f) {
        for (CL_INT r = get_global_id(0); r < NUM_POINTS; r += get_global_size(0)) {
            CL_INT label = g_labels[r];
            CL_FP point = g_points[ccoord2ind(NUM_POINTS, r, g_feature_base + f)];
            l_centroids[
                block_offset + ccoord2ind(NUM_CLUSTERS, label, l_feature_base + f)
            ] += point;
        }
    }

    for (CL_INT f = 0; f < NUM_THREAD_FEATURES; ++f) {
        for (CL_INT c = 0; c < NUM_CLUSTERS; ++c) {
            CL_INT mass = g_mass[c];
            CL_FP centroid = l_centroids[
                block_offset + ccoord2ind(NUM_CLUSTERS, c, l_feature_base + f)
            ];
            centroid = centroid / mass;
            g_centroids[
                g_cluster_offset + ccoord2ind(NUM_CLUSTERS, c, g_feature_base + f)
            ] = centroid;
        }
    }
}
