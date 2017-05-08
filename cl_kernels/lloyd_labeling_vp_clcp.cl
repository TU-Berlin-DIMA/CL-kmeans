/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 * 
 * 
 * Copyright (c) 2016-2017, Lutz, Clemens <lutzcle@cml.li>
 */

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
            __local VEC_TYPE(CL_POINT) *const restrict l_points,
            const CL_INT NUM_POINTS,
            const CL_INT NUM_CLUSTERS
       ) {

    for (
            CL_INT p = get_global_id(0) * VEC_LEN;
            p < NUM_POINTS - VEC_LEN + 1;
            p += get_global_size(0) * VEC_LEN
        )
    {

        // Phase 1
        VEC_TYPE(CL_LABEL) label = VLOAD(&g_labels[p]);

        for (CL_INT f = 0; f < NUM_FEATURES; ++f) {
            VEC_TYPE(CL_POINT) point =
                VLOAD(&g_points[ccoord2ind(NUM_POINTS, p, f)]);

            l_points[ccoord2ind(
                    get_local_size(0),
                    get_local_id(0),
                    f
                    )] = point;
        }

        VEC_TYPE(CL_LABEL) min_c;
        VEC_TYPE(CL_POINT) min_dist = CL_POINT_MAX;

        for (CL_LABEL c = 0; c < NUM_CLUSTERS; ++c) {
            VEC_TYPE(CL_POINT) dist = 0;

            for (CL_INT f = 0; f < NUM_FEATURES; ++f) {

                VEC_TYPE(CL_POINT) point =
                    l_points[ccoord2ind(
                            get_local_size(0),
                            get_local_id(0),
                            f
                            )];

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
