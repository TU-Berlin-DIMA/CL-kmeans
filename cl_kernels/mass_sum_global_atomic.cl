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
void mass_sum_global_atomic(
            __global CL_INT *const g_labels,
            __global CL_INT *const g_mass,
            const CL_INT NUM_POINTS,
            const CL_INT NUM_CLUSTERS
       ) {

    for (CL_INT r = 0; r < NUM_CLUSTERS; r += get_global_size(0)) {
        CL_INT c = r + get_local_id(0);

        if (c < NUM_CLUSTERS) {
            g_mass[c] = 0;
        }
    }

    for (CL_INT r = 0; r < NUM_POINTS; r += get_global_size(0)) {
        // Current point ID
        CL_INT p = r + get_local_id(0);

        if (p < NUM_POINTS) {
            CL_INT cluster = g_labels[p];
#ifdef TYPE32
            atomic_inc(&g_mass[cluster]);
#else
#ifdef TYPE64
            atom_inc(&g_mass[cluster]);
#endif
#endif
        }
    }
}
