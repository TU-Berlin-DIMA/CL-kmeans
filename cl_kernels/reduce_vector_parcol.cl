/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 * 
 * 
 * Copyright (c) 2016-2018, Lutz, Clemens <lutzcle@cml.li>"
 */

#ifdef TYPE32
#define CL_INT uint
#else
#ifdef TYPE64
#define CL_INT ulong
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
#endif
#endif

/* Reduce to single vector
 * Assume NUM_COLS * NUM_ROWS < 2 * local_size
 * Assume NUM_ROWS % local_size == 0
 * Assume local_size == global_size
 */
__kernel
void reduce_vector_parcol_inner(
        __global CL_TYPE *const restrict g_data,
        CL_INT const NUM_COLS,
        CL_INT const NUM_ROWS
        ) {

    __local CL_TYPE l_data[2 * WORKGROUP_SIZE];

    CL_INT base = get_group_id(0) * get_local_size(0);

    event_t event;
    async_work_group_copy(l_data, &g_data[base], 2 * get_local_size(0), event);
    wait_group_events(1, &event);
    barrier(CLK_LOCAL_MEM_FENCE);

    for (
            CL_INT remaining = get_local_size(0);
            remaining >= NUM_ROWS;
            remaining /= 2
            ) {
        if (get_local_id(0) < remaining) {
            l_data[get_local_id(0)] += l_data[remaining + get_local_id(0)];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    async_work_group_copy(&g_data[base], l_data, NUM_ROWS, event);
    wait_group_events(1, &event);
}


/*
 * Partially reduce multiple vectors
 * Assume column-major format
 * Assume num_rows < global_size
 * Assume global_size % num_rows == 0
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
