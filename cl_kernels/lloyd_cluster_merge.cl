/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright (c) 2016, Lutz, Clemens <lutzcle@cml.li>
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

__kernel
void lloyd_cluster_merge(
        __global CL_POINT const *const restrict g_points,
        __global CL_POINT *const restrict g_centroids,
        __global CL_MASS const *const restrict g_masses,
        __global CL_LABEL const *const restrict g_labels,
        __local CL_POINT *const restrict l_centroids,
        CL_INT const NUM_FEATURES,
        CL_INT const NUM_POINTS,
        CL_INT const NUM_CLUSTERS
        )
{
    CL_INT const l_cluster_offset =
        get_local_id(0)
        * NUM_FEATURES
        * NUM_CLUSTERS;
    CL_INT const g_cluster_offset =
        get_global_id(0)
        * NUM_FEATURES
        * NUM_CLUSTERS;

    for (CL_INT f = 0; f < NUM_FEATURES; ++f) {
        for (CL_INT c = 0; c < NUM_CLUSTERS; ++c) {
            l_centroids[
                l_cluster_offset + ccoord2ind(NUM_CLUSTERS, c, f)
            ] = 0;
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for (CL_INT f = 0; f < NUM_FEATURES; ++f) {
        for (CL_INT r = get_global_id(0); r < NUM_POINTS; r += get_global_size(0)) {
            CL_LABEL label = g_labels[r];
            CL_POINT point = g_points[ccoord2ind(NUM_POINTS, r, f)];

            l_centroids[
                l_cluster_offset + ccoord2ind(NUM_CLUSTERS, label, f)
            ] += point;
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for (CL_INT f = 0; f < NUM_FEATURES; ++f) {
        for (CL_INT c = 0; c < NUM_CLUSTERS; ++c) {
            CL_MASS mass = g_masses[c];
            CL_POINT centroid = l_centroids[
                l_cluster_offset + ccoord2ind(NUM_CLUSTERS, c, f)
            ];
            centroid = centroid / mass;
            g_centroids[
                g_cluster_offset + ccoord2ind(NUM_CLUSTERS, c, f)
            ] = centroid;
        }
    }
}
