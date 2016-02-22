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

#if VEC_LEN == 1
#define CL_FPVEC float
#define CL_INTVEC uint
#define CL_SINTVEC int
#define CL_VLOAD vload1
#define CL_VSTORE vstore1
#endif

#if VEC_LEN == 2
#define CL_FPVEC float2
#define CL_INTVEC uint2
#define CL_SINTVEC int2
#define CL_VLOAD vload2
#define CL_VSTORE vstore2
#endif

#if VEC_LEN == 4
#define CL_FPVEC float4
#define CL_INTVEC uint4
#define CL_SINTVEC int4
#define CL_VLOAD vload4
#define CL_VSTORE vstore4
#endif

#if VEC_LEN == 8
#define CL_FPVEC float8
#define CL_INTVEC uint8
#define CL_SINTVEC int8
#define CL_VLOAD vload8
#define CL_VSTORE vstore8
#endif

#else

#ifdef TYPE64
#define CL_FP double
#define CL_INT ulong
#define CL_FP_MAX DBL_MAX

#if VEC_LEN == 1
#define CL_FPVEC double
#define CL_INTVEC ulong
#define CL_SINTVEC long
#endif

#if VEC_LEN == 2
#define CL_FPVEC double2
#define CL_INTVEC ulong2
#define CL_SINTVEC long2
#define CL_VLOAD vload2
#define CL_VSTORE vstore2
#endif

#if VEC_LEN == 4
#define CL_FPVEC double4
#define CL_INTVEC ulong4
#define CL_SINTVEC long4
#define CL_VLOAD vload4
#define CL_VSTORE vstore4
#endif

#if VEC_LEN == 8
#define CL_FPVEC double8
#define CL_INTVEC ulong8
#define CL_SINTVEC long8
#define CL_VLOAD vload8
#define CL_VSTORE vstore8
#endif

#endif
#endif

CL_INT ccoord2ind(CL_INT rdim, CL_INT row, CL_INT col) {
    return rdim * col + row;
}

CL_INT rcoord2ind(CL_INT cdim, CL_INT row, CL_INT col) {
    return cdim * row + col;
}

__kernel
void lloyd_labeling_vp_clc(
            __global char *g_did_changes,
            __global CL_FP const *const restrict g_points,
            __global CL_FP const *const restrict g_centroids,
            __global CL_INT *const restrict g_labels,
            __local CL_FP *const restrict l_centroids,
            const CL_INT NUM_FEATURES,
            const CL_INT NUM_POINTS,
            const CL_INT NUM_CLUSTERS
       ) {

    bool did_changes = false;

    // Assume centroids fit into local memory

    // Read to local memory
    for (CL_INT i = get_local_id(0); i < NUM_CLUSTERS; i += get_local_size(0)) {
        for (CL_INT f = 0; f < NUM_FEATURES; ++f) {
            l_centroids[ccoord2ind(NUM_CLUSTERS, get_local_id(0), f)]
                = g_centroids[ccoord2ind(NUM_CLUSTERS, get_local_id(0), f)];
        }
    }

	barrier(CLK_LOCAL_MEM_FENCE);

    for (CL_INT p = get_global_id(0) * VEC_LEN; p < NUM_POINTS; p += get_global_size(0) * VEC_LEN) {

        // Phase 1
#if VEC_LEN == 1
        CL_INTVEC label = g_labels[p];
#else
        CL_INTVEC label = CL_VLOAD(0, &g_labels[p]);
#endif

        CL_INTVEC min_c;
        CL_FPVEC min_dist = CL_FP_MAX;

        for (CL_INT d = 0; d < NUM_CLUSTERS; d += CLUSTERS_UNROLL) {
#pragma unroll CLUSTERS_UNROLL
            for (CL_INT c = 0; c < CLUSTERS_UNROLL; ++c) {
                CL_FPVEC dist = 0;

#pragma unroll FEATURES_UNROLL
                for (CL_INT f = 0; f < FEATURES_UNROLL; ++f) {
#if VEC_LEN == 1
                    CL_FPVEC point =
                        g_points[ccoord2ind(NUM_POINTS, p, f)];
#else
                    CL_FPVEC point =
                        CL_VLOAD(
                                0,
                                &g_points[ccoord2ind(NUM_POINTS, p, f)]
                                );
#endif
                    CL_FPVEC difference =
                        point - l_centroids[
                        ccoord2ind(NUM_CLUSTERS, d + c, f)
                        ];

                    dist += difference * difference;
                }

                CL_SINTVEC is_dist_smaller = isless(dist, min_dist);
                min_dist = select(min_dist, dist, is_dist_smaller);
                min_c = select(min_c, d + c, is_dist_smaller);
            }
        }

#if VEC_LEN == 1
        g_labels[p] = min_c;
        did_changes |= min_c != label;
#else
        CL_VSTORE(min_c, 0, &g_labels[p]);
        did_changes |= any(min_c != label);
#endif
    }

    // Write back to global memory
    if (did_changes == true) {
        *g_did_changes = true;
    }
}
