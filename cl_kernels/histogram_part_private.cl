/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 * 
 * 
 * Copyright (c) 2016-2017, Lutz, Clemens <lutzcle@cml.li>
 */

// #define UNIT_STRIDE // i.e. sequential
// #define LOCAL_STRIDE
// Default: global stride access

#ifndef CL_INT
#define CL_INT uint
#endif
#ifndef CL_TYPE_IN
#define CL_TYPE_IN uint
#endif
#ifndef CL_TYPE_OUT
#define CL_TYPE_OUT uint
#endif

__kernel
void histogram_part_private(
            __global CL_TYPE_IN const *const restrict g_in,
            __global CL_TYPE_OUT *const restrict g_out,
            __local CL_TYPE_OUT *const restrict local_buf,
            const CL_INT NUM_ITEMS,
            const CL_INT NUM_BINS
       )
{
    CL_INT const local_offset = get_local_id(0) * NUM_BINS;
    CL_INT const group_bins = get_local_size(0) * NUM_BINS;
    CL_INT const group_offset = get_group_id(0) * group_bins;

    for (CL_INT c = 0; c < NUM_BINS; ++c) {
        local_buf[local_offset + c] = 0;
    }

#ifdef UNIT_STRIDE
    CL_INT block_size =
        (NUM_ITEMS + get_global_size(0) - 1) / get_global_size(0);
    CL_INT start_offset = get_global_id(0) * block_size;
    CL_INT real_block_size = (start_offset + block_size > NUM_ITEMS)
        ? sub_sat(NUM_ITEMS, start_offset)
        : block_size
        ;

    for (
            CL_INT p = start_offset;
            p < start_offset + real_block_size;
            ++p
        )
#else
#ifdef LOCAL_STRIDE
    CL_INT stride = get_local_size(0);
    CL_INT block_size =
        (NUM_ITEMS + get_num_groups(0) - 1) / get_num_groups(0);
    CL_INT group_start_offset = get_group_id(0) * block_size;
    CL_INT start_offset = group_start_offset + get_local_id(0);
    CL_INT real_block_size =
        (group_start_offset + block_size > NUM_ITEMS)
        ? sub_sat(NUM_ITEMS, group_start_offset)
        : block_size
        ;

    for (
            CL_INT p = start_offset;
            p < group_start_offset + real_block_size;
            p += stride
        )
#else
    for (
            CL_INT p = get_global_id(0);
            p < NUM_ITEMS;
            p += get_global_size(0)
        )
#endif
#endif
    {
        CL_TYPE_IN cluster = g_in[p];
        local_buf[local_offset + cluster] += 1;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    event_t event;
    async_work_group_copy(
            &g_out[group_offset],
            local_buf,
            group_bins,
            event
            );
}
