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

#define AIRITY 2

__kernel
void reduce_vector_divisor_parcol_innersum(
        __global CL_TYPE *const restrict g_data,
        CL_INT const NUM_COLS,
        CL_INT const NUM_ROWS,
        CL_INT const ROUND
        ) {

    CL_INT base = get_group_id(0) * get_local_size(0) * 2;
    CL_INT summand_a = base << ROUND;
    CL_INT summand_b = (base + get_local_size(0)) << ROUND;

    CL_TYPE sum = g_data[summand_a] + g_data[summand_b];

    g_data[summand_a] = sum;
}


/*
 * Partially reduce multiple vectors
 * Assume column-major format
 * Assume num_rows < global_size
 */
__kernel
void reduce_vector_parcol_compact(
        __global CL_TYPE *const restrict g_data,
        CL_INT const N
        ) {

    CL_TYPE sum = 0;
    for (CL_INT i = get_global_id(0); i < N; i += get_global_size(0)) {
        sum += g_data[i];
    }
    g_data[get_global_id(0)] = sum;

}
