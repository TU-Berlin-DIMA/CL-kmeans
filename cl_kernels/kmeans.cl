/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 * 
 * 
 * Copyright (c) 2016, Lutz, Clemens <lutzcle@cml.li>
 */

ulong ccoord2ind(ulong dim, ulong row, ulong col) {
    return dim * col + row;
}

ulong rcoord2ind(ulong dim, ulong row, ulong col) {
    return dim * row + col;
}

__kernel
void kmeans_with_host(
            __global char *g_did_changes,
            __global double const *const g_points,
            __global double const *const g_centroids,
            __global ulong *const g_memberships,
            __local double *const l_centroids,
            __local double *const l_old_centroids,
            const ulong NUM_FEATURES,
            const ulong NUM_POINTS,
            const ulong NUM_CLUSTERS
       ) {

	const size_t gid = get_global_id(0);
	const size_t lid = get_local_id(0);
	const size_t WORK_GROUP_SIZE = get_local_size(0);
	const size_t WORK_ITEM_SIZE = NUM_POINTS / get_global_size(0);
	const ulong LOCAL_NUM_POINTS = NUM_POINTS / get_num_groups(0);

    bool did_changes = false;

    // Assume centroids fit into local memory

    // Read to local memory
    for (ulong i = 0; i < NUM_CLUSTERS; i += WORK_GROUP_SIZE) {
        if (i + lid < NUM_CLUSTERS) {
            for (ulong f = 0; f < NUM_FEATURES; ++f) {
                l_old_centroids[ccoord2ind(NUM_FEATURES, i + lid, f)]
                    = g_centroids[ccoord2ind(NUM_FEATURES, i + lid, f)];
            }
        }
    }
	barrier(CLK_LOCAL_MEM_FENCE);

    for (ulong r = 0; r < NUM_POINTS; r += get_global_size(0)) {

        // Currently processing point number
        ulong p = gid + r;

        if (p < NUM_POINTS) {
            // Phase 1
            ulong membership = g_memberships[p];

            ulong min_c;
            double min_dist = DBL_MAX;

            for (ulong c = 0; c < NUM_CLUSTERS; ++c) {
                double dist = 0;
                for (ulong f = 0; f < NUM_FEATURES; ++f) {
                    double point = g_points[ccoord2ind(NUM_FEATURES, p, f)];
                    double difference = point - l_old_centroids[ccoord2ind(NUM_FEATURES, c, f)];
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
