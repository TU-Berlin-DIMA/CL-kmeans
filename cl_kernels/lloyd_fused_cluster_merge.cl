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
#define CL_POINT_MAX DBL_MAX
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
void lloyd_fused_cluster_merge(
        __global CL_POINT const *const restrict g_points,
        __global CL_POINT *const restrict g_centroids,
        __global CL_MASS *const restrict g_masses,
        __global CL_LABEL const *const restrict g_labels,
        __local CL_POINT *const restrict l_points,
        __local CL_POINT *const restrict l_old_centroids,
        __local CL_POINT *const restrict l_new_centroids,
        __local CL_MASS *const restrict l_masses,
        CL_INT const NUM_FEATURES,
        CL_INT const NUM_POINTS,
        CL_INT const NUM_CLUSTERS
        )
{

    // Calculate centroids offset
    CL_INT const l_cluster_offset =
        get_local_id(0)
        * NUM_FEATURES
        * NUM_CLUSTERS;
    CL_INT const g_cluster_offset =
        get_global_id(0)
        * NUM_FEATURES
        * NUM_CLUSTERS;

    // Calculate masses offset
    CL_INT const l_masses_offset =
        get_local_id(0) * NUM_CLUSTERS;
    CL_INT const g_masses_offset =
        get_global_id(0) * NUM_CLUSTERS;

    // Zero new centroids in local memory
    for (CL_INT f = 0; f < NUM_FEATURES; ++f) {
        for (CL_INT c = 0; c < NUM_CLUSTERS; ++c) {
            l_new_centroids[
                l_cluster_offset + ccoord2ind(NUM_CLUSTERS, c, f)
            ] = 0;
        }
    }

    // Zero masses in local memory
    for (CL_INT c = 0; c < NUM_CLUSTERS; ++c) {
        l_masses[l_masses_offset + c] = 0;
    }

    // Cache old centroids
    for (CL_INT f = 0; f < NUM_FEATURES; ++f) {
        for (
                CL_INT c = get_local_id(0);
                c < NUM_CLUSTERS;
                c += get_local_size(0))
        {
            l_old_centroids[ccoord2ind(NUM_CLUSTERS, c, f)]
                = g_centroids[ccoord2ind(NUM_CLUSTERS, c, f)];

        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for (
            CL_INT p = get_global_id(0);
            p < NUM_POINTS;
            p += get_global_size(0))
    {
        // Cache current point
        for (CL_INT f = 0; f < NUM_FEATURES; ++f) {
            // Read point
            CL_POINT point
            = g_points[ccoord2ind(NUM_POINTS, p, f)];

            // Cache point
            l_points[
                ccoord2ind(get_local_size(0), get_local_id(0), f)
            ] = point;
        }

        // Labeling phase
        CL_LABEL label;
        CL_POINT min_dist = CL_POINT_MAX;

        for (CL_INT c = 0; c < NUM_CLUSTERS; ++c) {

            CL_POINT dist = 0;

            for (CL_INT f = 0; f < NUM_FEATURES; ++f) {
                // Read point
                CL_POINT point
                    = l_points[
                    ccoord2ind(get_local_size(0), get_local_id(0), f)
                    ];

                // Calculate distance
                CL_POINT difference
                    = point - l_old_centroids[
                    ccoord2ind(NUM_CLUSTERS, c, f)
                    ];
                dist += difference * difference;
            }

            bool is_dist_smaller = dist < min_dist;
            min_dist = is_dist_smaller ? dist : min_dist;
            label = is_dist_smaller ? c : label;
        }

        // Masses update phase
        l_masses[l_masses_offset + label] += 1;

        // Centroids update phase
        for (CL_INT f = 0; f < NUM_FEATURES; ++f) {
            CL_POINT point
                = l_points[
                ccoord2ind(get_local_size(0), get_local_id(0), f)
                ];

            l_new_centroids[
                l_cluster_offset + ccoord2ind(NUM_CLUSTERS, label, f)
            ] += point;

        }

    }

    // No barrier necessary, as only writing back private data

    for (CL_INT c = 0; c < NUM_CLUSTERS; ++c) {
        // Write back masses
        CL_MASS mass = l_masses[l_masses_offset + c];
        g_masses[g_masses_offset + c] = mass;

        // Write back centroids
        for (CL_INT f = 0; f < NUM_FEATURES; ++f) {
            CL_POINT centroid = l_new_centroids[
                l_cluster_offset + ccoord2ind(NUM_CLUSTERS, c, f)
            ];
            g_centroids[
                g_cluster_offset + ccoord2ind(NUM_CLUSTERS, c, f)
            ] = centroid;
        }
    }
}
