/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 * 
 * 
 * Copyright (c) 2016-2017, Lutz, Clemens <lutzcle@cml.li>
 */

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

    for (
            CL_INT p = get_global_id(0);
            p < NUM_ITEMS;
            p += get_global_size(0)
        )
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
