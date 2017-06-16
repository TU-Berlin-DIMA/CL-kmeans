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
#ifndef CL_TYPE_IN
#define CL_TYPE_IN uint
#endif
#ifndef CL_TYPE_OUT
#define CL_TYPE_OUT uint
#endif

#ifndef VEC_LEN
#define VEC_LEN 1
#else
#define VEC_TYPE_JUMP(TYPE, LEN) TYPE##LEN
#define VEC_TYPE_JUMP_2(TYPE, LEN) VEC_TYPE_JUMP(TYPE, LEN)
#define VEC_TYPE(TYPE) VEC_TYPE_JUMP_2(TYPE, VEC_LEN)

#define VLOAD_JUMP(I, P, LEN) vload##LEN(I, P)
#define VLOAD_JUMP_2(I, P, LEN) VLOAD_JUMP(I, P, LEN)
#define VLOAD(I, P) VLOAD_JUMP_2(I, P, VEC_LEN)
#endif

CL_INT ccoord2ind(CL_INT dim, CL_INT row, CL_INT col) {
    return dim * col + row;
}

CL_INT rcoord2ind(CL_INT dim, CL_INT row, CL_INT col) {
    return dim * row + col;
}

__kernel
void histogram_part_private(
            __global CL_TYPE_IN const *const restrict g_in,
            __global CL_TYPE_OUT *const restrict g_out,
#ifndef GLOBAL_MEM
            __local CL_TYPE_OUT *const restrict local_buf,
#endif
            const CL_INT NUM_ITEMS,
            const CL_INT NUM_BINS
       )
{
    CL_INT const group_bins = get_local_size(0) * NUM_BINS;
    CL_INT const group_offset = get_group_id(0) * group_bins;

    // Zero bins
    for (CL_INT c = 0; c < NUM_BINS; ++c) {
#ifdef GLOBAL_MEM
        g_out[group_offset + rcoord2ind(NUM_BINS, get_local_id(0), c)] = 0;
#else
        local_buf[ccoord2ind(get_local_size(0), get_local_id(0), c)] = 0;
#endif
    }

    CL_INT p;
#ifdef LOCAL_STRIDE
    CL_INT stride = VEC_LEN * get_local_size(0);
    CL_INT block_size =
        (NUM_ITEMS + get_num_groups(0) - 1) / get_num_groups(0);
    block_size = block_size - block_size % VEC_LEN;
    CL_INT group_start_offset = get_group_id(0) * block_size;
    CL_INT start_offset = group_start_offset + VEC_LEN * get_local_id(0);
    CL_INT real_block_size =
        (group_start_offset + block_size > NUM_ITEMS)
        ? sub_sat(NUM_ITEMS, group_start_offset)
        : block_size
        ;

    for (
            p = start_offset;
            p < group_start_offset + real_block_size;
            p += stride
        )
#else
    for (
            p = VEC_LEN * get_global_id(0);
            p < NUM_ITEMS - VEC_LEN + 1;
            p += VEC_LEN * get_global_size(0)
        )
#endif
    {
#if VEC_LEN > 1
#ifdef GLOBAL_MEM
#define BASE_STEP(NUM)                                                \
        g_out[                                                        \
            group_offset +                                            \
            rcoord2ind(NUM_BINS, get_local_id(0), cluster.s ## NUM )  \
        ] += 1;
#else
#define BASE_STEP(NUM)                                                    \
        local_buf[                                                        \
        ccoord2ind(get_local_size(0), get_local_id(0), cluster.s ## NUM ) \
        ] += 1;
#endif

#define REP_STEP_2 BASE_STEP(0) BASE_STEP(1)
#define REP_STEP_4 REP_STEP_2 BASE_STEP(2) BASE_STEP(3)
#define REP_STEP_8 REP_STEP_4 BASE_STEP(4) BASE_STEP(5)         \
        BASE_STEP(6) BASE_STEP(7)
#define REP_STEP_16 REP_STEP_8 BASE_STEP(8) BASE_STEP(9)        \
        BASE_STEP(a) BASE_STEP(b) BASE_STEP(c) BASE_STEP(d) \
        BASE_STEP(e) BASE_STEP(f)
#define REP_STEP_JUMP(NUM) REP_STEP_ ## NUM
#define REP_STEP(NUM) do { REP_STEP_JUMP(NUM) } while (false)

        VEC_TYPE(CL_TYPE_IN) cluster = VLOAD(0, &g_in[p]);
        REP_STEP(VEC_LEN);
#else

        CL_TYPE_IN cluster = g_in[p];
#ifdef GLOBAL_MEM
        g_out[group_offset + rcoord2ind(NUM_BINS, get_local_id(0), cluster)] += 1;
#else
        local_buf[ccoord2ind(get_local_size(0), get_local_id(0), cluster)] += 1;
#endif
#endif
    }

#ifndef GLOBAL_MEM
    for (CL_INT c = 0; c < NUM_BINS; ++c) {
        CL_TYPE_OUT cluster =
            local_buf[ccoord2ind(get_local_size(0), get_local_id(0), c)];
        g_out[group_offset + rcoord2ind(NUM_BINS, get_local_id(0), c)] =
            cluster;
    }
#endif
}
