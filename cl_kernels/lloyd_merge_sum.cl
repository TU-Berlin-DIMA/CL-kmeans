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
        const CL_INT NUM_FEATURES,
        const CL_INT NUM_POINTS,
        const CL_INT NUM_CLUSTERS
       ) {

    CL_INT const my_row = get_global_id(0) % NUM_CLUSTERS;
    CL_INT const my_col = (get_global_id(0) / NUM_CLUSTERS) % NUM_FEATURES;
    CL_INT const my_blk = (get_global_id(0) / NUM_CLUSTERS) / NUM_FEATURES;
    CL_INT const blk_size = NUM_CLUSTERS * NUM_FEATURES;
    CL_INT const num_blks = (blk_size + get_global_size(0) - 1) / get_global_size(0);
    CL_INT const num_local_rows = (get_local_size(0) + NUM_CLUSTERS - 1) / NUM_CLUSTERS;

    if (my_col < NUM_FEATURES) {
        CL_FP centroid = 0;

        for (CL_INT p = get_global_id(0); p < NUM_POINTS; p += blk_size) {
            // p: Current point global index

            if (my_row < NUM_CLUSTERS) {
                l_points[get_local_id(0)] = g_points[p];
            }

            barrier(CLK_LOCAL_MEM_FENCE);

            for (CL_INT r = 0; r < num_local_rows; ++r) {
                centroid += l_points[ccoord2ind(num_local_rows, r, my_col)];
            }
        }

        // TODO: only 1 thread per centroid loads mass
        // threads processing other features of same centroid use it
        if (my_row < NUM_CLUSTERS) {
            CL_INT mass = g_mass[my_col];
            g_centroids[get_global_id(0)] = centroid / mass;
        }
    }
}
