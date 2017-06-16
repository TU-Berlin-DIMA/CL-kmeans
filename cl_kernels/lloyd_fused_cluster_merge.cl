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

// Anti-bank conflict column major indexing
// Warning: Use only for local memory buffers
CL_INT ccoord2abc(CL_INT dim, CL_INT row, CL_INT col) {
    return get_local_size(0) * (dim * col + row) + get_local_id(0);
}

// Note: Define NUM_FEATURES in preprocessor
__kernel
void lloyd_fused_cluster_merge(
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
        CL_INT const NUM_POINTS,
        CL_INT const NUM_CLUSTERS
        )
{

    // Calculate centroids offset
    CL_INT const g_cluster_offset =
        get_global_id(0)
        * NUM_FEATURES
        * NUM_CLUSTERS;

    // Calculate masses offset
    CL_INT const g_masses_offset =
        get_global_id(0) * NUM_CLUSTERS;

#ifdef GLOBAL_MEM
    // Zero new centroids in global memory
    for (CL_INT c = 0; c < NUM_CLUSTERS; ++c) {
        for (CL_INT f = 0; f < NUM_FEATURES; ++f) {
            g_new_centroids[
                g_cluster_offset + ccoord2ind(NUM_CLUSTERS, c, f)
            ] = 0;
        }
    }

    // Zero masses in global memory
    for (CL_INT c = 0; c < NUM_CLUSTERS; ++c) {
        g_masses[g_masses_offset + c] = 0;
    }
#else
    // Zero new centroids in local memory
    for (CL_INT f = 0; f < NUM_FEATURES; ++f) {
        for (CL_INT c = 0; c < NUM_CLUSTERS; ++c) {
            l_new_centroids[
                ccoord2abc(NUM_CLUSTERS, c, f)
            ] = 0;
        }
    }

    // Zero masses in local memory
    for (CL_INT c = 0; c < NUM_CLUSTERS; ++c) {
        l_masses[ccoord2ind(get_local_size(0), get_local_id(0), c)] = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);
#endif

    CL_INT p;
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
            p = start_offset;
            p < group_start_offset + real_block_size;
            p += stride
        )
#else
    for (
            p = get_global_id(0) * VEC_LEN;
            p < NUM_POINTS - VEC_LEN + 1;
            p += get_global_size(0) * VEC_LEN
        )
#endif
    {
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

        for (CL_INT c = 0; c < NUM_CLUSTERS; ++c) {

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
        VSTORE(label, &g_labels[p]);

        // Masses update phase
#if VEC_LEN > 1
#ifdef GLOBAL_MEM
#define MASS_INC_BASE(NUM)                                               \
        g_masses[g_masses_offset + label.s ## NUM ] += 1;
#else
#define MASS_INC_BASE(NUM)                                               \
        l_masses[                                                        \
        ccoord2ind(get_local_size(0), get_local_id(0), label.s ## NUM )  \
        ] += 1;
#endif

        REP_STEP(MASS_INC_BASE, VEC_LEN);
#else
#ifdef GLOBAL_MEM
        g_masses[g_masses_offset + label] += 1;
#else
        l_masses[ccoord2ind(get_local_size(0), get_local_id(0), label)] += 1;
#endif
#endif

        // Centroids update phase
        for (CL_INT f = 0; f < NUM_FEATURES; ++f) {
            VEC_TYPE(CL_POINT) point =
#ifdef GLOBAL_MEM
                VLOAD(&g_points[ccoord2ind(NUM_POINTS, p, f)]);
#else
                l_points[
                    ccoord2ind(get_local_size(0), get_local_id(0), f)
                ];
#endif

#if VEC_LEN > 1
#ifdef GLOBAL_MEM
#define CENTROID_UPDATE_BASE(NUM)                                              \
            g_new_centroids[                                                   \
                g_cluster_offset + ccoord2ind(NUM_CLUSTERS, label.s ## NUM, f) \
            ] += point.s ## NUM;
#else
#define CENTROID_UPDATE_BASE(NUM)                               \
            l_new_centroids[                                    \
                ccoord2abc(NUM_CLUSTERS, label.s ## NUM, f)     \
            ] += point.s ## NUM;
#endif

            REP_STEP(CENTROID_UPDATE_BASE, VEC_LEN);
#else
#ifdef GLOBAL_MEM
            g_new_centroids[
                g_cluster_offset + ccoord2ind(NUM_CLUSTERS, label, f)
            ] += point;
#else
            l_new_centroids[
                ccoord2abc(NUM_CLUSTERS, label, f)
            ] += point;
#endif
#endif

        }

    }

#ifndef GLOBAL_MEM
    // No barrier necessary, as only writing back private data

    for (CL_INT c = 0; c < NUM_CLUSTERS; ++c) {
        // Write back masses
        CL_MASS mass = l_masses[ccoord2ind(get_local_size(0), get_local_id(0), c)];
        g_masses[g_masses_offset + c] = mass;

        // Write back centroids
        for (CL_INT f = 0; f < NUM_FEATURES; ++f) {
            CL_POINT centroid = l_new_centroids[
                ccoord2abc(NUM_CLUSTERS, c, f)
            ];
            g_new_centroids[
                g_cluster_offset + ccoord2ind(NUM_CLUSTERS, c, f)
            ] = centroid;
        }
    }
#endif
}
