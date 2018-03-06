/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright (c) 2016-2018, Lutz, Clemens <lutzcle@cml.li>"
 */

// #define LOCAL_STRIDE
// Default: global stride access
//
// #define GLOBAL_MEM
// Default: local memory cache

#ifndef CL_INT
#define CL_INT uint
#endif

// float -> int ; double -> long
#ifndef CL_SINT
#define CL_SINT int
#endif

#ifndef CL_POINT
#define CL_POINT float
#endif

#ifndef CL_LABEL
#define CL_LABEL uint
#endif

#ifndef CL_MASS
#define CL_MASS uint
#endif

#ifndef CL_POINT_MAX
#define CL_POINT_MAX FLT_MAX
#endif

#ifndef VEC_LEN
#define VEC_LEN 1
#endif

#if VEC_LEN == 1
#define VEC_TYPE(TYPE) TYPE
#define VLOAD(P) (*(P))
#define VSTORE(DATA, P) do { *(P) = DATA; } while (false)

#else
#define VEC_TYPE_JUMP(TYPE, LEN) TYPE##LEN
#define VEC_TYPE_JUMP_2(TYPE, LEN) VEC_TYPE_JUMP(TYPE, LEN)
#define VEC_TYPE(TYPE) VEC_TYPE_JUMP_2(TYPE, VEC_LEN)

#define VLOAD_JUMP(P, LEN) vload##LEN(0, P)
#define VLOAD_JUMP_2(P, LEN) VLOAD_JUMP(P, LEN)
#define VLOAD(P) VLOAD_JUMP_2(P, VEC_LEN)

#define VSTORE_JUMP(DATA, P, LEN) vstore##LEN(DATA, 0, P)
#define VSTORE_JUMP_2(DATA, P, LEN) VSTORE_JUMP(DATA, P, LEN)
#define VSTORE(DATA, P) VSTORE_JUMP_2(DATA, P, VEC_LEN)
#endif

#define REP_STEP_2(BASE_STEP) BASE_STEP(0) BASE_STEP(1)
#define REP_STEP_4(BASE_STEP) REP_STEP_2(BASE_STEP)                 \
    BASE_STEP(2) BASE_STEP(3)
#define REP_STEP_8(BASE_STEP) REP_STEP_4(BASE_STEP)                 \
    BASE_STEP(4) BASE_STEP(5)                                       \
    BASE_STEP(6) BASE_STEP(7)
#define REP_STEP_16(BASE_STEP) REP_STEP_8(BASE_STEP)                \
    BASE_STEP(8) BASE_STEP(9) BASE_STEP(a) BASE_STEP(b)             \
    BASE_STEP(c) BASE_STEP(d) BASE_STEP(e) BASE_STEP(f)
#define REP_STEP_JUMP(BASE_STEP, NUM) REP_STEP_ ## NUM (BASE_STEP)
#define REP_STEP(BASE_STEP, NUM)                                    \
do { REP_STEP_JUMP(BASE_STEP, NUM) } while (false)

CL_INT ccoord2ind(CL_INT dim, CL_INT row, CL_INT col) {
    return dim * col + row;
}

// Note: Define NUM_FEATURES with preprocessor
__kernel
void lloyd_fused_feature_sum(
        __global CL_POINT const *const restrict g_points,
        __constant CL_POINT const *const restrict g_old_centroids,
        __global CL_POINT *const restrict g_new_centroids,
        __global CL_MASS *const restrict g_masses,
        __global CL_LABEL *const restrict g_labels,
#ifndef GLOBAL_MEM
        __local VEC_TYPE(CL_POINT) *const restrict l_points,
        __local CL_POINT *const restrict l_new_centroids,
        __local CL_MASS *const restrict l_masses,
#endif
        __local VEC_TYPE(CL_LABEL) *const restrict l_labels,
        CL_INT const NUM_POINTS,
        CL_INT const NUM_CLUSTERS,
        CL_INT const NUM_THREAD_FEATURES
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
    CL_INT const block_points_offset = block * num_block_points;
    CL_INT const block_offset = NUM_CLUSTERS * NUM_FEATURES * block;
    CL_INT const l_feature = get_local_id(0) % block_size;

    // Calculate tile offset
    CL_INT const tile_offset = NUM_CLUSTERS * NUM_FEATURES
        * (get_group_id(0) * num_blocks + block);

    // Calculate masses offset
    CL_INT const g_masses_offset =
        get_global_id(0) * NUM_CLUSTERS;

#ifdef GLOBAL_MEM
    // Zero new centroids in global memory
    for (
            CL_INT i = l_feature;
            i < NUM_FEATURES * NUM_CLUSTERS;
            i += block_size
        )
    {
        g_new_centroids[tile_offset + i] = 0;
    }

    // Zero masses in global memory
    for (CL_INT c = 0; c < NUM_CLUSTERS; ++c) {
        g_masses[g_masses_offset + c] = 0;
    }
#else
    // Zero new centroids in local memory
    for (
            CL_INT i = get_local_id(0);
            i < num_blocks * NUM_FEATURES * NUM_CLUSTERS;
            i += get_local_size(0))
    {
        l_new_centroids[i] = 0;
    }

    // Zero masses in local memory
    for (CL_INT c = 0; c < NUM_CLUSTERS; ++c) {
        l_masses[ccoord2ind(get_local_size(0), get_local_id(0), c)] = 0;
    }
#endif

    // Main loop over points
    //
    // All threads must participate!
    // This is because we must handle thread work item assignment
    // separately for each phase
#ifdef LOCAL_STRIDE
    CL_INT iter_stride = VEC_LEN * get_local_size(0);
    CL_INT iter_block_size =
        (NUM_POINTS + get_num_groups(0) - 1) / get_num_groups(0);
    iter_block_size = iter_block_size - iter_block_size % VEC_LEN;
    CL_INT group_start_offset = get_group_id(0) * iter_block_size;
    CL_INT real_iter_block_size =
        (group_start_offset + iter_block_size > NUM_POINTS)
        ? sub_sat(NUM_POINTS, group_start_offset)
        : iter_block_size
        ;

    for (
            CL_INT group_offset = group_start_offset;
            group_offset < group_start_offset + real_iter_block_size;
            group_offset += iter_stride
        )
#else
    for (
            CL_INT group_offset = get_group_id(0) * get_local_size(0) * VEC_LEN;
            group_offset < NUM_POINTS - VEC_LEN + 1;
            group_offset += get_global_size(0) * VEC_LEN
        )
#endif
    {
        CL_INT p = group_offset + get_local_id(0) * VEC_LEN;

        // In 1st iteration, wait for zero-ed buffers and old centroids
        // In 2nd iteration, wait for feature sums from previous iteration
#ifdef GLOBAL_MEM
        barrier(CLK_GLOBAL_MEM_FENCE);
#else
        barrier(CLK_LOCAL_MEM_FENCE);
#endif

        if (p < NUM_POINTS - VEC_LEN + 1) {

#ifndef GLOBAL_MEM
            // Cache current point
            for (CL_INT f = 0; f < NUM_FEATURES; ++f) {
                // Read point
                VEC_TYPE(CL_POINT) point
                    = VLOAD(&g_points[ccoord2ind(NUM_POINTS, p, f)]);

                // Cache point
                l_points[
                    ccoord2ind(get_local_size(0), get_local_id(0), f)
                ] = point;
            }
#endif

            // Labeling phase
            VEC_TYPE(CL_LABEL) label;
            VEC_TYPE(CL_POINT) min_dist = CL_POINT_MAX;

            for (CL_LABEL c = 0; c < NUM_CLUSTERS; ++c) {

                VEC_TYPE(CL_POINT) dist = 0;

                for (CL_INT f = 0; f < NUM_FEATURES; ++f) {
                    // Read point
                    VEC_TYPE(CL_POINT) point =
#ifdef GLOBAL_MEM
                        VLOAD(&g_points[ccoord2ind(NUM_POINTS, p, f)]);
#else
                        l_points[
                        ccoord2ind(get_local_size(0), get_local_id(0), f)
                        ];
#endif

                    // Calculate distance
                    VEC_TYPE(CL_POINT) difference
                        = point - g_old_centroids[
                        ccoord2ind(NUM_CLUSTERS, c, f)
                        ];
                    dist = fma(difference, difference, dist);
                }

                VEC_TYPE(CL_SINT) is_dist_smaller = isless(dist, min_dist);
                min_dist = fmin(dist, min_dist);
                label = select(label, c, is_dist_smaller);
            }

            // Write back label
            l_labels[get_local_id(0)] = label;
            VSTORE(label, &g_labels[p]);

            // Masses update phase
#if VEC_LEN > 1
#ifdef GLOBAL_MEM
#define MASS_INC_BASE(NUM)                                               \
            g_masses[g_masses_offset + label.s ## NUM ] += 1;
#else
#define MASS_INC_BASE(NUM)                                               \
            l_masses[ccoord2ind(                                         \
                    get_local_size(0),                                   \
                    get_local_id(0),                                     \
                    label.s ## NUM                                       \
                    )] += 1;
#endif

            REP_STEP(MASS_INC_BASE, VEC_LEN);
#else
#ifdef GLOBAL_MEM
            g_masses[g_masses_offset + label] += 1;
#else
            l_masses[ccoord2ind(get_local_size(0), get_local_id(0), label)] += 1;
#endif
#endif

        }

        // Wait for labels
        barrier(CLK_LOCAL_MEM_FENCE);

        // Calculate the number of points to process in CU phase
        //
        // Usually this is num_local_points
        // In case when local size is not a divisor of NUM_POINTS,
        // this doesn't hold and we need to get real number.
        CL_INT g_block_points_offset =
            group_offset + block_points_offset * VEC_LEN;

        CL_INT num_real_block_points =
            (
             g_block_points_offset + num_block_points * VEC_LEN
             >= NUM_POINTS
            )
            ? sub_sat(NUM_POINTS, (g_block_points_offset)) / VEC_LEN
            : num_block_points;

        // Centroids update phase
        for (
                CL_INT bp = block_points_offset;
                bp < block_points_offset + num_real_block_points;
                bp += 1)
        {
            VEC_TYPE(CL_LABEL) label = l_labels[bp];

            for (
                    CL_INT f = l_feature;
                    f < NUM_FEATURES;
                    f += block_size
                )
            {
                VEC_TYPE(CL_POINT) point =
#ifdef GLOBAL_MEM
                    VLOAD(&g_points[ccoord2ind(NUM_POINTS, group_offset + bp, f)]);
#else
                    l_points[
                    ccoord2ind(num_local_points, bp, f)
                    ];
#endif

#if VEC_LEN > 1
#ifdef GLOBAL_MEM
#define CENTROID_UPDATE_BASE(NUM)                                        \
                g_new_centroids[                                         \
                        tile_offset                                      \
                        + ccoord2ind(NUM_CLUSTERS, label.s ## NUM, f)    \
                ] += point.s ## NUM;
#else
#define CENTROID_UPDATE_BASE(NUM)                                        \
                l_new_centroids[ccoord2ind(                              \
                        NUM_FEATURES * num_blocks,                       \
                        NUM_FEATURES * block + f,                        \
                        label.s ## NUM                                   \
                        )] += point.s ## NUM;
#endif

                REP_STEP(CENTROID_UPDATE_BASE, VEC_LEN);
#else
#ifdef GLOBAL_MEM
                g_new_centroids[
                    tile_offset + ccoord2ind(NUM_CLUSTERS, label, f)
                ] += point;
#else
                l_new_centroids[ccoord2ind(
                        NUM_FEATURES * num_blocks,
                        NUM_FEATURES * block + f,
                        label
                        )] += point;
#endif
#endif
            }
        }

    }

#ifndef GLOBAL_MEM
    barrier(CLK_LOCAL_MEM_FENCE);

    for (CL_INT c = 0; c < NUM_CLUSTERS; ++c) {
        // Write back masses
        CL_MASS mass = l_masses[ccoord2ind(get_local_size(0), get_local_id(0), c)];
        g_masses[g_masses_offset + c] = mass;

        // Write back centroids
        for (
                CL_INT f = l_feature;
                f < NUM_FEATURES;
                f += block_size
            )
        {
            CL_POINT centroid = l_new_centroids[ccoord2ind(
                    NUM_FEATURES * num_blocks,
                    NUM_FEATURES * block + f,
                    c
                    )];
            g_new_centroids[
                tile_offset + ccoord2ind(NUM_CLUSTERS, c, f)
            ] = centroid;
        }
    }
#endif

}
