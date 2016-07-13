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

#define CASE_STMT(X) case (X): ++p_bins[X]; break;

#define CASE_REPL_2(BASE) CASE_STMT(BASE) CASE_STMT(BASE+1)
#define CASE_REPL_4(BASE) CASE_REPL_2(BASE) CASE_REPL_2(BASE+2)
#define CASE_REPL_8(BASE) CASE_REPL_4(BASE) CASE_REPL_4(BASE+4)
#define CASE_REPL_16(BASE) CASE_REPL_8(BASE) CASE_REPL_8(BASE+8)

#define CASE_REPL_E(NUM) CASE_REPL_##NUM(0)
#define CASE_REPL(NUM) CASE_REPL_E(NUM)

/*
 * Calculate histogram in paritions per work group
 * 
 * g_in: global_size
 * g_out: num_work_groups * NUM_BINS
 *
 * NUM_BINS defined as power of 2 at compile time
 */
__kernel
void histogram_part_private(
            __global CL_INT const *const restrict g_in,
            __global CL_INT *const restrict g_out,
            const CL_INT NUM_ITEMS
       ) {

    CL_INT p_bins[NUM_BINS] = {};

    for (
            CL_INT r = get_global_id(0);
            r < NUM_BINS * get_num_groups(0);
            r += get_global_size(0)
            ) {
        g_out[r] = 0;
    }

    for (CL_INT r = 0; r < NUM_ITEMS; r += get_global_size(0)) {
        // Current point ID
        CL_INT p = r + get_global_id(0);

        if (p < NUM_ITEMS) {
            CL_INT cluster = g_in[p];

            switch (cluster) {
                CASE_REPL(NUM_BINS);
            }
        }
    }

    barrier(CLK_GLOBAL_MEM_FENCE);

    CL_INT group_offset = get_group_id(0) * NUM_BINS;

#pragma unroll NUM_BINS
    for (CL_INT bin = 0; bin < NUM_BINS; ++bin) {
        __global CL_INT *g_bin = &g_out[group_offset + bin];
        CL_INT val = p_bins[bin];
#ifdef TYPE32
        atomic_add(g_bin, val);
#else
#ifdef TYPE64
        atom_add(g_bin, val);
#endif
#endif
    }
}
