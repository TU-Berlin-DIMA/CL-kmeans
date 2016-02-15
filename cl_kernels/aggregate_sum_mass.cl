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
 * Aggregation sum over mass_sum_merge output
 * i.e. aggregate rows of column-major matrix
 *
 * g_mass: NUM_BLOCKS x NUM_CLUSTERS
 */
__kernel
void aggregate_sum_mass(
        __global CL_INT *const restrict g_mass,
        CL_INT const NUM_CLUSTERS,
        CL_INT const NUM_BLOCKS
        ) {

    for (CL_INT cluster_offset = get_global_id(0);
            cluster_offset < NUM_CLUSTERS;
            cluster_offset += get_global_size(0)) {

        CL_INT mass = 0;

        for (CL_INT block = 0;
                block < NUM_BLOCKS * NUM_CLUSTERS;
                block += NUM_CLUSTERS) {

            mass += g_mass[block + cluster_offset];
        }

        g_mass[cluster_offset] = mass;
    }
}
