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

CL_INT ccoord2ind(CL_INT dim, CL_INT row, CL_INT col) {
    return dim * col + row;
}

CL_INT rcoord2ind(CL_INT dim, CL_INT row, CL_INT col) {
    return dim * row + col;
}

// Assumes local space:
// Centroid block: NUM_CLUSTERS x local_size
// Points block:   local_size x local_size
__kernel
void lloyd_feature_sum(
            __global CL_FP const *const g_points,
            __global CL_FP *const g_centroids,
            __global CL_INT *const g_labels,
            __local CL_FP *const l_centroids,
            __local CL_FP *const l_points,
            const CL_INT NUM_FEATURES,
            const CL_INT NUM_POINTS,
            const CL_INT NUM_CLUSTERS
       ) {

    for (CL_INT x = 0; x < NUM_FEATURES; x += get_global_size(0)) {
        // Initialize centroids to zero
        for (CL_INT c = 0; c < NUM_CLUSTERS; ++c) {
            l_centroids[ccoord2ind(NUM_CLUSTERS, c, get_local_id(0))] = 0;
        }

        // Process block of points
        for (CL_INT r = 0; r < NUM_POINTS; r += get_local_size(0)) {

            if (r + get_local_id(0) < NUM_POINTS) {
                // Coalesc access to points
                for (CL_INT w = 0; w < get_local_size(0) && x + w < NUM_FEATURES; ++w) {
                    l_points[ccoord2ind(get_local_size(0), get_local_id(0), w)] =
                        g_points[ccoord2ind(NUM_POINTS, r + get_local_id(0), x + w)];
                }
            }

            barrier(CLK_LOCAL_MEM_FENCE);

            // Process points in block
            if (x + get_global_id(0) < NUM_FEATURES) {
                for (CL_INT p = 0; p < get_local_size(0) && r + p < NUM_POINTS; ++p) {

                    // Coalesced access to labels
                    CL_INT label = g_labels[r + p];
                    CL_FP coord = l_points[ccoord2ind(get_local_size(0), p, get_local_id(0))];

                    l_centroids[ccoord2ind(NUM_CLUSTERS, label, get_local_id(0))] += coord;
                }
            }

            barrier(CLK_LOCAL_MEM_FENCE);
        }

        // Coalesc write back block of centroids
        for (CL_INT c = 0; c < NUM_CLUSTERS; c += get_local_size(0)) {
            if (c + get_local_id(0) < NUM_CLUSTERS) {
                for (CL_INT w = 0; w < get_local_size(0) && x + w < NUM_FEATURES; ++w) {
                    g_centroids[ccoord2ind(NUM_CLUSTERS, c + get_local_id(0), x + w)] =
                        l_centroids[ccoord2ind(NUM_CLUSTERS, c + get_local_id(0), w)];
                }
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }
}
