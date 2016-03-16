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

#define MAX_NUM_FEATURES 64

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
        CL_INT const NUM_FEATURES,
        CL_INT const NUM_POINTS,
        CL_INT const NUM_CLUSTERS
        ) {

    CL_INT const centroids_size = NUM_FEATURES * NUM_CLUSTERS;
    CL_INT const num_local_blocks = get_local_size(0) / centroids_size;
    CL_INT const l_block = get_local_id(0) / centroids_size;
    CL_INT const g_block = get_global_id(0) / centroids_size;
    CL_INT const l_pos = get_local_id(0) - l_block * centroids_size;
    CL_INT const l_cluster = l_pos % NUM_CLUSTERS;
    CL_INT const l_feature = l_pos / NUM_CLUSTERS;
    CL_INT const cache_num_points = get_local_size(0) / NUM_FEATURES;
    CL_INT const g_num_points = get_global_size(0) / NUM_FEATURES;
    CL_INT const l_num_points = cache_num_points / num_local_blocks;
    CL_INT const l_point_begin = l_block * l_num_points;

    CL_FP centroid = 0;

    for (CL_INT r = get_group_id(0) * cache_num_points; r < NUM_POINTS; r += g_num_points) {

        uint num_copy_points = (
                (r + cache_num_points - 1 < NUM_POINTS) ?
                cache_num_points
                :
                NUM_POINTS - r
                );

        for (
                CL_INT p = l_point_begin;
                p < l_point_begin + l_num_points && r + p < NUM_POINTS;
                ++p) {

            bool is_in_cluster = (g_labels[r + p] == l_cluster);
            CL_FP point = g_points[ccoord2ind(NUM_POINTS, r + p, l_feature)];
            centroid += (is_in_cluster ? point : 0);
        }
    }

    event_t mass_copy_event = 0;
    async_work_group_copy(
            &l_mass[0],
            &g_mass[0],
            NUM_CLUSTERS,
            mass_copy_event
            );
    wait_group_events(1, &mass_copy_event);

    CL_INT mass = l_mass[l_cluster];
    CL_INT global_ind = g_block * centroids_size + ccoord2ind(NUM_CLUSTERS, l_cluster, l_feature);
    g_centroids[global_ind] = (centroid / mass);
}

__kernel
void lloyd_merge_tiles() {
}
