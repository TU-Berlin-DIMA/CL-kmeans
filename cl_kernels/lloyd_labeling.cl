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
            __global CL_FP const *const g_points,
            __global CL_FP const *const g_centroids,
            __global CL_INT *const g_memberships,
            __local CL_FP *const l_centroids,
            const CL_INT NUM_FEATURES,
            const CL_INT NUM_POINTS,
            const CL_INT NUM_CLUSTERS
       ) {

	const CL_INT gid = get_global_id(0);
	const CL_INT lid = get_local_id(0);
	const CL_INT WORK_GROUP_SIZE = get_local_size(0);
	const CL_INT WORK_ITEM_SIZE = NUM_POINTS / get_global_size(0);
	const CL_INT LOCAL_NUM_POINTS = NUM_POINTS / get_num_groups(0);

    bool did_changes = false;

    // Assume centroids fit into local memory

    // Read to local memory
    for (CL_INT i = 0; i < NUM_CLUSTERS; i += get_local_size(0)) {
        if (i + lid < NUM_CLUSTERS) {
            for (CL_INT f = 0; f < NUM_FEATURES; ++f) {
                l_centroids[ccoord2ind(NUM_CLUSTERS, i + lid, f)]
                    = g_centroids[ccoord2ind(NUM_CLUSTERS, i + lid, f)];
            }
        }
    }
	barrier(CLK_LOCAL_MEM_FENCE);

    for (CL_INT r = 0; r < NUM_POINTS; r += get_global_size(0)) {

        // Currently processing point number
        CL_INT p = gid + r;

        if (p < NUM_POINTS) {
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

                if (dist < min_dist) {
                    min_dist = dist;
                    min_c = c;
                }
            }

            if (min_c != membership) {
                did_changes = true;
            }

            g_memberships[p] = min_c;
        }
    }


    // Write back to global memory
    if (did_changes == true) {
        // g_did_changes[gid] = did_changes;
        *g_did_changes = true;
    }
}
