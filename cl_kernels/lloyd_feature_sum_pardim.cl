/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 * 
 * 
 * Copyright (c) 2016-2017, Lutz, Clemens <lutzcle@cml.li>
 */

// #define LOCAL_STRIDE
// Default: global stride access

#ifndef CL_INT
#define CL_INT ulong
#endif

#ifndef CL_POINT
#define CL_POINT double
#endif

#ifndef CL_LABEL
#define CL_LABEL ulong
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

CL_INT ccoord2ind(CL_INT dim, CL_INT row, CL_INT col) {
    return dim * col + row;
}

// Anti-bank conflict column major indexing
// Warning: Use only for local memory buffers
CL_INT ccoord2abc(CL_INT dim, CL_INT row, CL_INT col) {
    return get_local_size(0) * get_local_size(1) * (dim * col + row)
        + get_local_id(0) * get_local_size(1) + get_local_id(1);
}

CL_INT rcoord2ind(CL_INT dim, CL_INT row, CL_INT col) {
    return dim * row + col;
}

CL_INT div_round_up(CL_INT dividend, CL_INT divisor) {
    return (dividend + divisor - 1) / divisor;
}

// Define NUM_THREAD_FEATURES in preprocessor
__kernel
void lloyd_feature_sum_pardim(
        __global CL_POINT const *const restrict g_points,
        __global CL_POINT *const restrict g_centroids,
        __global CL_LABEL const *const restrict g_labels,
        __local CL_POINT *const restrict l_centroids,
        CL_INT const NUM_FEATURES,
        CL_INT const NUM_POINTS,
        CL_INT const NUM_CLUSTERS
        ) {

    CL_INT block = get_local_id(0);
    CL_INT tile_row = get_global_id(0);
    CL_INT tile_col = get_global_id(1);
    CL_INT num_col_tiles = get_global_size(1);
    CL_INT g_feature_base = NUM_THREAD_FEATURES * get_global_id(1);
    CL_INT g_cluster_offset = (num_col_tiles * tile_row)
        * NUM_THREAD_FEATURES * NUM_CLUSTERS;

    for (CL_INT f = 0; f < NUM_THREAD_FEATURES; ++f) {
        for (CL_INT c = 0; c < NUM_CLUSTERS; ++c) {
            l_centroids[
                ccoord2abc(NUM_CLUSTERS, c, f)
            ] = 0.0;
        }
    }

    CL_INT r;
#ifdef LOCAL_STRIDE
    CL_INT stride = VEC_LEN * get_local_size(0);
    CL_INT block_size =
        (NUM_POINTS + get_num_groups(0) - 1) / get_num_groups(0);
    block_size = block_size - block_size % VEC_LEN;
    CL_INT group_start_offset = get_group_id(0) * block_size;
    CL_INT start_offset = group_start_offset + VEC_LEN * get_local_id(0);
    CL_INT real_block_size =
        (group_start_offset + block_size > NUM_POINTS)
        ? sub_sat(NUM_POINTS, group_start_offset)
        : block_size
        ;

    for (
            r = start_offset;
            r < group_start_offset + real_block_size;
            r += stride
        )
#else
    for (
            r = get_global_id(0) * VEC_LEN;
            r < NUM_POINTS - VEC_LEN + 1;
            r += get_global_size(0) * VEC_LEN
        )
#endif
    {

        /*
         * Load label from memory and sync work group.
         *
         * Sync ensures that small time variances in
         * local memory access when adding point do not
         * cause one warp to outrun the others.
         * If one warp is faster than the others,
         * loading labels will experience L1 cache misses,
         * causing a performance decrease on GPUs
         * (tested with Nvidia GTX 1080).
         *
         * Note that local fence is intentional,
         * as global mem fence introduces more latency;
         * no writes occur that could cause inconsisteny.
        */
        VEC_TYPE(CL_LABEL) label = VLOAD(&g_labels[r]);
        barrier(CLK_LOCAL_MEM_FENCE);

        for (CL_INT f = 0; f < NUM_THREAD_FEATURES; ++f) {
            VEC_TYPE(CL_POINT) point =
                VLOAD(&g_points[
                        ccoord2ind(NUM_POINTS, r, g_feature_base + f)
                ]);
#if VEC_LEN > 1
#define BASE_STEP(NUM)                                                   \
            l_centroids[                                                 \
                ccoord2abc(                                              \
                        NUM_CLUSTERS, label.s ## NUM, f                  \
                        )                                                \
            ] += point.s ## NUM;

#define REP_STEP_2 BASE_STEP(0) BASE_STEP(1)
#define REP_STEP_4 REP_STEP_2 BASE_STEP(2) BASE_STEP(3)
#define REP_STEP_8 REP_STEP_4 BASE_STEP(4) BASE_STEP(5)         \
        BASE_STEP(6) BASE_STEP(7)
#define REP_STEP_16 REP_STEP_8 BASE_STEP(8) BASE_STEP(9)        \
        BASE_STEP(a) BASE_STEP(b) BASE_STEP(c) BASE_STEP(d) \
        BASE_STEP(e) BASE_STEP(f)
#define REP_STEP_JUMP(NUM) REP_STEP_ ## NUM
#define REP_STEP(NUM) do { REP_STEP_JUMP(NUM) } while (false)

            REP_STEP(VEC_LEN);
#else
            l_centroids[
                ccoord2abc(NUM_CLUSTERS, label, f)
            ] += point;
#endif

        }
    }

    for (CL_INT f = 0; f < NUM_THREAD_FEATURES; ++f) {
        for (CL_INT c = 0; c < NUM_CLUSTERS; ++c) {
            CL_POINT centroid = l_centroids[
                ccoord2abc(NUM_CLUSTERS, c, f)
            ];
            g_centroids[
                g_cluster_offset + ccoord2ind(NUM_CLUSTERS, c, g_feature_base + f)
            ] = centroid;
        }
    }
}
