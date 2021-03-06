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

CL_INT ccoord2ind(CL_INT rdim, CL_INT row, CL_INT col) {
    return rdim * col + row;
}

CL_INT rcoord2ind(CL_INT cdim, CL_INT row, CL_INT col) {
    return cdim * row + col;
}

// Note: Define NUM_FEATURES with preprocessor
__kernel
void lloyd_labeling_vp_clcp(
            __global CL_POINT const *const restrict g_points,
            __constant CL_POINT const *const restrict g_centroids,
            __global CL_LABEL *const restrict g_labels,
#ifndef GLOBAL_MEM
            __local VEC_TYPE(CL_POINT) *const restrict l_points,
#endif
            const CL_INT NUM_POINTS,
            const CL_INT NUM_CLUSTERS
       ) {

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
            CL_INT p = get_global_id(0) * VEC_LEN;
            p < NUM_POINTS - VEC_LEN + 1;
            p += get_global_size(0) * VEC_LEN
        )
#endif
    {

#ifndef GLOBAL_MEM
        // Cache points in local memory
        for (CL_INT f = 0; f < NUM_FEATURES; ++f) {
            VEC_TYPE(CL_POINT) point =
                VLOAD(&g_points[ccoord2ind(NUM_POINTS, p, f)]);

            l_points[ccoord2ind(
                    get_local_size(0),
                    get_local_id(0),
                    f
                    )] = point;
        }
#endif

        VEC_TYPE(CL_LABEL) min_c;
        VEC_TYPE(CL_POINT) min_dist = CL_POINT_MAX;

        for (CL_LABEL c = 0; c < NUM_CLUSTERS; ++c) {
            VEC_TYPE(CL_POINT) dist = 0;

            for (CL_INT f = 0; f < NUM_FEATURES; ++f) {

                VEC_TYPE(CL_POINT) point =
#ifdef GLOBAL_MEM
                    VLOAD(&g_points[ccoord2ind(NUM_POINTS, p, f)]);
#else
                    l_points[ccoord2ind(
                            get_local_size(0),
                            get_local_id(0),
                            f
                            )];
#endif

                VEC_TYPE(CL_POINT) difference =
                    point - g_centroids[ccoord2ind(NUM_CLUSTERS, c, f)];

                dist = fma(difference, difference, dist);
            }

            VEC_TYPE(CL_SINT) is_dist_smaller = isless(dist, min_dist);
            min_dist = fmin(min_dist, dist);
            min_c = select(min_c, c, is_dist_smaller);
        }

        VSTORE(min_c, &g_labels[p]);
    }
}
