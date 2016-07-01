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
#else
#ifdef TYPE64
#define CL_FP double
#define CL_INT ulong
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
#endif
#endif

__kernel
void histogram_global(
            __global CL_INT *const g_in,
            __global CL_INT *const g_out,
            const CL_INT NUM_ITEMS,
            const CL_INT NUM_BINS
       ) {

    for (CL_INT r = 0; r < NUM_BINS; r += get_global_size(0)) {
        CL_INT c = r + get_global_id(0);

        if (c < NUM_BINS) {
            g_out[c] = 0;
        }
    }

    for (CL_INT r = 0; r < NUM_ITEMS; r += get_global_size(0)) {
        // Current point ID
        CL_INT p = r + get_global_id(0);

        if (p < NUM_ITEMS) {
            CL_INT cluster = g_in[p];
#ifdef TYPE32
            atomic_inc(&g_out[cluster]);
#else
#ifdef TYPE64
            atom_inc(&g_out[cluster]);
#endif
#endif
        }
    }
}
