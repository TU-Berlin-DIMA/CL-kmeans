/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 * 
 * 
 * Copyright (c) 2016, Lutz, Clemens <lutzcle@cml.li>
 */

#include "pair.h"

#define WORK_ITEM_SIZE 4

__kernel
void loop_join(__constant int *table_A, ulong table_A_size,
  __constant int *table_B, ulong table_B_size,
  __global uint *table_joined_partition_sizes,
  __global uint *offsets,
  __global pair *table_joined) {

  /* define and initialize private variables */
  ulong private_table_joined_partition_size = 0;
  ulong global_addr = 0, local_addr = 0;
  ulong i = 0, j = 0, k = 0;
#ifndef COUNT_RESULT_ROWS
  const ulong table_joined_partition_offset = offsets[get_global_id(0)];
  const ulong table_joined_partition_size = table_joined_partition_sizes[get_global_id(0)];
#endif

  /* loop join */
  k = 0;
  global_addr = get_global_id(0) * WORK_ITEM_SIZE;
  for (i = global_addr; i != global_addr + WORK_ITEM_SIZE; ++i) {
    for (j = 0; j != table_B_size; ++j) {
      if (table_A[i] == table_B[j]) {
#ifdef COUNT_RESULT_ROWS
        ++private_table_joined_partition_size;
#else
        table_joined[table_joined_partition_offset + k].first = i;
        table_joined[table_joined_partition_offset + k].second = j;
        ++k;
#endif
      }
    }
  }

  /* write result to global memory */
#ifdef COUNT_RESULT_ROWS
  table_joined_partition_sizes[get_global_id(0)] = private_table_joined_partition_size;
#endif

}
