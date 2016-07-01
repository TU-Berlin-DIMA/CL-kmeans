/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 * 
 * 
 * Copyright (c) 2016, Lutz, Clemens <lutzcle@cml.li>
 */

#ifdef TYPE32
#define CL_INT uint
#else
#ifdef TYPE64
#define CL_INT ulong
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
#endif
#endif

/*
 * Calculate histogram in paritions per work group
 * 
 * g_in: global_size
 * g_out: num_work_groups * NUM_BINS
 * l_bins: NUM_BINS
 */
__kernel
void histogram_part_local(
            __global CL_INT const *const restrict g_in,
            __global CL_INT *const restrict g_out,
            __local CL_INT *const restrict l_bins,
            const CL_INT NUM_ITEMS,
            const CL_INT NUM_BINS
       ) {

    for (CL_INT r = 0; r < NUM_BINS; r += get_local_size(0)) {
        CL_INT c = r + get_local_id(0);

        if (c < NUM_BINS) {
            l_bins[c] = 0;
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for (CL_INT r = 0; r < NUM_ITEMS; r += get_global_size(0)) {
        // Current point ID
        CL_INT p = r + get_global_id(0);

        if (p < NUM_ITEMS) {
            CL_INT cluster = g_in[p];
#ifdef TYPE32
            atomic_inc(&l_bins[cluster]);
#else
#ifdef TYPE64
            atom_inc(&l_bins[cluster]);
#endif
#endif
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    CL_INT group_offset = get_group_id(0) * NUM_BINS;
    for (CL_INT r = 0; r < NUM_BINS; r += get_local_size(0)) {
        CL_INT l_c = r + get_local_id(0);
        CL_INT g_c = group_offset + l_c;

        if (l_c < NUM_BINS) {
            g_out[g_c] = l_bins[l_c];
        }
    }
}
