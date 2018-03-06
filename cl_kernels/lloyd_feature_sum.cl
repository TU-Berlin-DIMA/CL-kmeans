/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright (c) 2016-2018, Lutz, Clemens <lutzcle@cml.li>"
 */

#ifndef CL_INT
#define CL_INT uint
#endif
#ifndef CL_POINT
#define CL_POINT float
#endif
#ifndef CL_MASS
#define CL_MASS uint
#endif
#ifndef CL_LABEL
#define CL_LABEL uint
#endif
#ifndef VEC_LEN
#define VEC_LEN 4
#endif

#define CONCAT_EXPANDED(NAME, LEN) NAME##LEN
#define CONCAT(NAME, LEN) CONCAT_EXPANDED(NAME, LEN)

#define CL_POINT_V CONCAT(CL_POINT, VEC_LEN)
#define CL_LABEL_V CONCAT(CL_LABEL, VEC_LEN)

CL_INT ccoord2ind(CL_INT dim, CL_INT row, CL_INT col) {
    return dim * col + row;
}

__kernel
void lloyd_feature_sum_sequential(
        __global CL_POINT const *const restrict g_points,
        __global CL_POINT *const restrict g_centroids,
        __global CL_LABEL const *const restrict g_labels,
        const CL_INT NUM_FEATURES,
        const CL_INT NUM_POINTS,
        const CL_INT NUM_CLUSTERS
        )
{
    CL_INT const feature = get_global_id(0) % NUM_FEATURES;
    CL_INT const point_block = get_global_id(0) / NUM_FEATURES;
    CL_INT const num_point_blocks = get_global_size(0) / NUM_FEATURES;
    CL_INT const num_local_points =
        (NUM_POINTS + num_point_blocks - 1) / num_point_blocks;
    CL_INT const num_real_local_points =
        (num_local_points * (point_block + 1) > NUM_POINTS)
        ? sub_sat(NUM_POINTS, (num_local_points * point_block))
        : num_local_points
        ;
    CL_INT const point_offset = num_local_points * point_block;
    CL_INT const centroid_offset =
        NUM_CLUSTERS * NUM_FEATURES * point_block;
    CL_INT const feature_offset =
        centroid_offset + NUM_CLUSTERS * feature;

    for (
            CL_INT i = feature_offset;
            i < feature_offset + NUM_CLUSTERS;
            ++i
        )
    {
        g_centroids[i] = 0;
    }

    for (
            CL_INT p = point_offset;
            p < point_offset + num_real_local_points;
            ++p
        )
    {

        CL_LABEL label = g_labels[p];
        CL_POINT point = g_points[ccoord2ind(NUM_POINTS, p, feature)];

        g_centroids[
            centroid_offset + ccoord2ind(NUM_CLUSTERS, label, feature)
        ] += point;
    }
}

__kernel
void lloyd_feature_sum_sequential_v(
        __global CL_POINT const *const restrict g_points,
        __global CL_POINT *const restrict g_centroids,
        __global CL_LABEL const *const restrict g_labels,
        const CL_INT NUM_FEATURES,
        const CL_INT NUM_POINTS,
        const CL_INT NUM_CLUSTERS
        )
{
    CL_INT const feature = get_global_id(0) % NUM_FEATURES;
    CL_INT const point_block = get_global_id(0) / NUM_FEATURES;
    CL_INT const num_point_blocks = get_global_size(0) / NUM_FEATURES;
    CL_INT const num_local_points =
        (NUM_POINTS + num_point_blocks - 1) / num_point_blocks;
    CL_INT const num_real_local_points =
        (num_local_points * (point_block + 1) > NUM_POINTS)
        ? sub_sat(NUM_POINTS, (num_local_points * point_block))
        : num_local_points
        ;
    CL_INT const point_offset = num_local_points * point_block;
    CL_INT const centroid_offset =
        NUM_CLUSTERS * NUM_FEATURES * point_block;
    CL_INT const feature_offset =
        centroid_offset + NUM_CLUSTERS * feature;

    for (
            CL_INT i = feature_offset;
            i < feature_offset + NUM_CLUSTERS;
            ++i
        )
    {
        g_centroids[i] = 0;
    }

    CL_INT p;
    for (
            p = point_offset;
            (p + VEC_LEN - 1) < (point_offset + num_real_local_points);
            p += VEC_LEN
        )
    {

        CL_LABEL_V label = *(
                (__global CL_LABEL_V const * const)
                (g_labels + p)
                );
        CL_POINT_V point = *(
                (__global CL_POINT_V const * const)
                (g_points + ccoord2ind(NUM_POINTS, p, feature))
                );

#define ADD_POINT(OFFSET)                                   \
        g_centroids[                                        \
        centroid_offset +                                   \
        ccoord2ind(NUM_CLUSTERS, label.s##OFFSET , feature) \
        ] += point.s##OFFSET

        ADD_POINT(0);
        ADD_POINT(1);
#if VEC_LEN >= 4
        ADD_POINT(2);
        ADD_POINT(3);
#endif
#if VEC_LEN >= 8
        ADD_POINT(4);
        ADD_POINT(5);
        ADD_POINT(6);
        ADD_POINT(7);
#endif
    }

    // if (p != point_offset + num_real_local_points) {
    // printf("p: %d, limit: %d [%d]\n", p, point_offset + num_real_local_points, get_global_id(0));
    //     for (
    //             p -= VEC_LEN;
    //             p < point_offset + num_real_local_points;
    //             ++p
    //         )
    //     {
    //         CL_LABEL label = g_labels[p];
    //         CL_POINT point = g_points[ccoord2ind(NUM_POINTS, p, feature)];
    //
    //         g_centroids[
    //             centroid_offset + ccoord2ind(NUM_CLUSTERS, label, feature)
    //         ] += point;
    //     }
    // }
}
