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
 * Calculate sum of cluster masses
 * 
 * g_labels: global_size
 * g_mass: num_work_groups * NUM_CLUSTERS
 * l_mass: NUM_CLUSTERS
 */
__kernel
void mass_sum_merge(
            __global CL_INT const *const restrict g_labels,
            __global CL_INT *const restrict g_mass,
            __local CL_INT *const restrict l_mass,
            const CL_INT NUM_POINTS,
            const CL_INT NUM_CLUSTERS
       ) {

    for (CL_INT r = 0; r < NUM_CLUSTERS; r += get_local_size(0)) {
        CL_INT c = r + get_local_id(0);

        if (c < NUM_CLUSTERS) {
            l_mass[c] = 0;
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for (CL_INT r = 0; r < NUM_POINTS; r += get_global_size(0)) {
        // Current point ID
        CL_INT p = r + get_global_id(0);

        if (p < NUM_POINTS) {
            CL_INT cluster = g_labels[p];
#ifdef TYPE32
            atomic_inc(&l_mass[cluster]);
#else
#ifdef TYPE64
            atom_inc(&l_mass[cluster]);
#endif
#endif
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    CL_INT group_offset = get_group_id(0) * NUM_CLUSTERS;
    for (CL_INT r = 0; r < NUM_CLUSTERS; r += get_local_size(0)) {
        CL_INT l_c = r + get_local_id(0);
        CL_INT g_c = group_offset + l_c;

        if (l_c < NUM_CLUSTERS) {
            g_mass[g_c] = l_mass[l_c];
        }
    }
}
