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

    CL_INT l_feature = get_local_id(1);
    CL_INT l_num_features = get_local_size(1);
    CL_INT l_block = get_local_id(0);
    CL_INT l_num_blocks = get_local_size(0);
    CL_INT l_offset = l_num_features * NUM_CLUSTERS * l_block;
    CL_INT g_feature = get_global_id(1);
    CL_INT g_tile = get_group_id(0);

    for (CL_INT c = 0; c < NUM_CLUSTERS; ++c) {
        l_centroids[
            l_offset + ccoord2ind(NUM_CLUSTERS, c, l_feature)
        ] = 0;
    }

    for (CL_INT r = get_global_id(0); r < NUM_POINTS; r += get_global_size(0)) {
        CL_INT label = g_labels[r];
        CL_FP point = g_points[ccoord2ind(NUM_POINTS, r, g_feature)];
        l_centroids[
            l_offset + ccoord2ind(NUM_CLUSTERS, label, l_feature)
        ] += point;
    }

    CL_INT g_clusters_base = NUM_CLUSTERS * NUM_FEATURES * l_num_blocks * g_tile;
    for (CL_INT c = 0; c < NUM_CLUSTERS; ++c) {
        CL_INT mass = g_mass[c];
        CL_FP centroid = l_centroids[
            l_offset + ccoord2ind(NUM_CLUSTERS, c, l_feature)
        ];
        centroid = centroid / mass;
        g_centroids[
            g_clusters_base + ccoord2ind(NUM_CLUSTERS, c, g_feature)
        ] = centroid;
    }
}
