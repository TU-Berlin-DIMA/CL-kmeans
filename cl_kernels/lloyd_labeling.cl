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
#define CL_FP_MAX FLT_MAX
#else
#ifdef TYPE64
#define CL_FP double
#define CL_INT ulong
#define CL_FP_MAX DBL_MAX
#endif
#endif

CL_INT ccoord2ind(CL_INT rdim, CL_INT row, CL_INT col) {
    return rdim * col + row;
}

CL_INT rcoord2ind(CL_INT cdim, CL_INT row, CL_INT col) {
    return cdim * row + col;
}

__kernel
void lloyd_labeling(
            __global char *g_did_changes,
            __global CL_FP const *const restrict g_points,
            __global CL_FP const *const restrict g_centroids,
            __global CL_INT *const restrict g_memberships,
            __local CL_FP *const restrict l_centroids,
            const CL_INT NUM_FEATURES,
            const CL_INT NUM_POINTS,
            const CL_INT NUM_CLUSTERS
       ) {

    bool did_changes = false;

    // Assume centroids fit into local memory

    // Read to local memory
    for (CL_INT i = get_local_id(0); i < NUM_CLUSTERS; i += get_local_size(0)) {
        for (CL_INT f = 0; f < NUM_FEATURES; ++f) {
            l_centroids[ccoord2ind(NUM_CLUSTERS, i, f)]
                = g_centroids[ccoord2ind(NUM_CLUSTERS, i, f)];
        }
    }

	barrier(CLK_LOCAL_MEM_FENCE);

    for (CL_INT p = get_global_id(0); p < NUM_POINTS; p += get_global_size(0)) {

        // Phase 1
        CL_INT membership = g_memberships[p];

        CL_INT min_c;
        CL_FP min_dist = CL_FP_MAX;

        for (CL_INT c = 0; c < NUM_CLUSTERS; ++c) {
            CL_FP dist = 0;
            for (CL_INT f = 0; f < NUM_FEATURES; ++f) {
                CL_FP point = g_points[ccoord2ind(NUM_POINTS, p, f)];
                CL_FP difference = point - l_centroids[ccoord2ind(NUM_CLUSTERS, c, f)];
                dist += difference * difference;
            }

            bool is_dist_smaller = dist < min_dist;
            min_dist = (is_dist_smaller ? dist : min_dist);
            min_c = (is_dist_smaller ? c : min_c);
        }

        did_changes |= (min_c != membership ? true : false);

        g_memberships[p] = min_c;
    }


    // Write back to global memory
    if (did_changes == true) {
        // g_did_changes[gid] = did_changes;
        *g_did_changes = true;
    }
}
