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

__kernel
void lloyd_feature_sum(
            __global CL_FP const *const g_points,
            __global CL_FP *const g_centroids,
            __global CL_INT *const g_labels,
            __local CL_FP *const l_centroids,
            const CL_INT NUM_FEATURES,
            const CL_INT NUM_POINTS,
            const CL_INT NUM_CLUSTERS
       ) {

	const CL_INT WORK_GROUP_SIZE = get_local_size(0);
	const CL_INT WORK_ITEM_SIZE = NUM_POINTS / get_global_size(0);
	const CL_INT LOCAL_NUM_POINTS = NUM_POINTS / get_num_groups(0);

    // My feature ID
    CL_INT const g_f = get_global_id(0);

    // Local feature offset
    CL_INT const l_f = get_local_id(0);

    if (g_f < NUM_FEATURES) {
        for (CL_INT c = 0; c < NUM_CLUSTERS; ++c) {
            l_centroids[ccoord2ind(NUM_CLUSTERS, c, l_f)] = 0;
        }

        for (CL_INT p = 0; p < NUM_POINTS; ++p) {

            CL_INT label = g_labels[p];
            CL_FP coord = g_points[ccoord2ind(NUM_POINTS, p, g_f)];

            l_centroids[ccoord2ind(NUM_CLUSTERS, label, l_f)] += coord;
        }

        for (CL_INT c = 0; c < NUM_CLUSTERS; ++c) {
            g_centroids[ccoord2ind(NUM_CLUSTERS, c, g_f)]
                = l_centroids[ccoord2ind(NUM_CLUSTERS, c, l_f)];
        }
    }
}
