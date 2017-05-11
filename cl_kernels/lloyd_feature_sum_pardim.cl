/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 * 
 * 
 * Copyright (c) 2016-2017, Lutz, Clemens <lutzcle@cml.li>
 */

#ifndef CL_INT
#define CL_INT ulong
#endif

#ifndef CL_POINT
#define CL_POINT double
#endif

#ifndef CL_LABEL
#define CL_LABEL ulong
#endif

#ifndef CL_MASS
#define CL_MASS ulong
#endif

CL_INT ccoord2ind(CL_INT dim, CL_INT row, CL_INT col) {
    return dim * col + row;
}

// Anti-bank conflict column major indexing
// Warning: Use only for local memory buffers
CL_INT ccoord2abc(CL_INT dim, CL_INT row, CL_INT col) {
    return get_local_size(0) * (dim * col + row) + get_local_id(0);
}

CL_INT rcoord2ind(CL_INT dim, CL_INT row, CL_INT col) {
    return dim * row + col;
}

CL_INT div_round_up(CL_INT dividend, CL_INT divisor) {
    return (dividend + divisor - 1) / divisor;
}

// Define NUM_THREAD_FEATURES in preprocessor
__kernel
void lloyd_feature_sum_pardim(
        __global CL_POINT const *const restrict g_points,
        __global CL_POINT *const restrict g_centroids,
        __global CL_MASS const *const restrict g_mass,
        __global CL_LABEL const *const restrict g_labels,
        __local CL_POINT *const restrict l_centroids,
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
            ] = 0.0;
        }
    }

    for (CL_INT r = get_global_id(0); r < NUM_POINTS; r += get_global_size(0)) {
        CL_LABEL label = g_labels[r];
        for (CL_INT f = 0; f < NUM_THREAD_FEATURES; ++f) {
            CL_POINT point = g_points[ccoord2ind(NUM_POINTS, r, g_feature_base + f)];
            l_centroids[
                block_offset + ccoord2ind(NUM_CLUSTERS, label, l_feature_base + f)
            ] += point;
        }
    }

    for (CL_INT f = 0; f < NUM_THREAD_FEATURES; ++f) {
        for (CL_INT c = 0; c < NUM_CLUSTERS; ++c) {
            CL_MASS mass = g_mass[c];
            CL_POINT centroid = l_centroids[
                block_offset + ccoord2ind(NUM_CLUSTERS, c, l_feature_base + f)
            ];
            centroid = centroid / mass;
            g_centroids[
                g_cluster_offset + ccoord2ind(NUM_CLUSTERS, c, g_feature_base + f)
            ] = centroid;
        }
    }
}
