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
void lloyd_fused_feature_sum(
        __global CL_POINT const *const restrict g_points,
        __global CL_POINT const *const restrict g_old_centroids,
        __global CL_POINT *const restrict g_new_centroids,
        __global CL_MASS *const restrict g_masses,
        __global CL_LABEL *const restrict g_labels,
        __local CL_POINT *const restrict l_points,
        __local CL_POINT *const restrict l_old_centroids,
        __local CL_POINT *const restrict l_new_centroids,
        __local CL_MASS *const restrict l_masses,
        __local CL_LABEL *const restrict l_labels,
        CL_INT const NUM_FEATURES,
        CL_INT const NUM_POINTS,
        CL_INT const NUM_CLUSTERS
        )
{

    // Calculate centroids indices
    CL_INT const block_size = NUM_FEATURES / NUM_THREAD_FEATURES;
    CL_INT const block =
        get_local_id(0) / block_size;
    CL_INT const num_local_points = get_local_size(0);
    CL_INT const num_blocks =
        get_local_size(0) / block_size;
    CL_INT const num_block_points =
        num_local_points / num_blocks;
    CL_INT const tile =
        get_global_id(0) / block_size;
    CL_INT const block_offset = NUM_CLUSTERS * NUM_FEATURES * block;
    CL_INT const tile_offset = NUM_CLUSTERS * NUM_FEATURES * (get_group_id(0) * num_blocks + block);
    CL_INT const l_feature = get_local_id(0) % block_size;

    // Calculate masses offset
    CL_INT const l_masses_offset =
        get_local_id(0) * NUM_CLUSTERS;
    CL_INT const g_masses_offset =
        get_global_id(0) * NUM_CLUSTERS;

    // Zero new centroids in local memory
    for (
            CL_INT i = get_local_id(0);
            i < get_local_size(0) * NUM_THREAD_FEATURES * NUM_CLUSTERS;
            i += get_local_size(0))
    {
        l_new_centroids[i] = 0;
    }

    // Zero masses in local memory
    for (CL_INT c = 0; c < NUM_CLUSTERS; ++c) {
        l_masses[l_masses_offset + c] = 0;
    }

    // Cache old centroids
    event_t centroids_cache_event;
    async_work_group_copy(
            l_old_centroids,
            g_old_centroids,
            NUM_CLUSTERS * NUM_FEATURES,
            centroids_cache_event
            );
    wait_group_events(1, &centroids_cache_event);

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

        // Write back label
        l_labels[get_local_id(0)] = label;
        g_labels[p] = label;


        // Masses update phase
        l_masses[l_masses_offset + label] += 1;

        barrier(CLK_LOCAL_MEM_FENCE);

        // Centroids update phase
        for (
                CL_INT bp = block * num_block_points;
                bp < (block + 1) * num_block_points;
                bp += 1)
        {
            for (
                    CL_INT f = l_feature * NUM_THREAD_FEATURES;
                    f < (l_feature + 1) * NUM_THREAD_FEATURES;
                    f += 1)
            {
                label = l_labels[bp];

                CL_POINT point
                    = l_points[
                    ccoord2ind(num_local_points, bp, f)
                    ];

                l_new_centroids[
                    block_offset
                        + ccoord2ind(NUM_CLUSTERS, label, f)
                ] += point;
            }
        }

    }

    barrier(CLK_LOCAL_MEM_FENCE);

    event_t mass_event;
    async_work_group_copy(
            &g_masses[
            NUM_CLUSTERS * get_group_id(0) * get_local_size(0)
            ],
            l_masses,
            NUM_CLUSTERS * get_local_size(0),
            mass_event
            );

    event_t centroid_event;
    async_work_group_copy(
            &g_new_centroids[
            NUM_CLUSTERS * NUM_FEATURES * get_group_id(0) * num_blocks
            ],
            l_new_centroids,
            NUM_CLUSTERS * NUM_FEATURES * num_blocks,
            centroid_event
            );
    wait_group_events(1, &mass_event);
    wait_group_events(1, &centroid_event);
}
