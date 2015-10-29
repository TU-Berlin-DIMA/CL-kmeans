#define NULL ((void *) 0)

__kernel 
void prefix_sum(
		__global uint *const g_idata,
		__global uint *const g_odata,
		__global uint *const g_carry,
		__local uint *const l_data,
		const ulong DATA_SIZE
		) {

	const size_t gid = get_global_id(0);
	const size_t lid = get_local_id(0);
	const size_t WORK_GROUP_SIZE = get_local_size(0);
	const size_t WORK_ITEM_SIZE = DATA_SIZE / get_global_size(0);
	const ulong LOCAL_DATA_SIZE = DATA_SIZE / get_num_groups(0);

	ulong ind_a = 0;
	ulong ind_b = 0;
	ulong d = 0;
	ulong work_group_bound = 0;
	uint tmp = 0;

	l_data[2*lid] = g_idata[2*gid];
	l_data[2*lid +1] = g_idata[2*gid + 1];

	barrier(CLK_LOCAL_MEM_FENCE);

	work_group_bound = WORK_GROUP_SIZE;
	for (d = 1; d != LOCAL_DATA_SIZE; d = d * 2) {
		if (lid < work_group_bound) {
			ind_a = 2*d*lid + d - 1;
			ind_b = 2*d*lid + 2*d - 1;
			l_data[ind_b] = l_data[ind_a] + l_data[ind_b];
		}

		work_group_bound = work_group_bound / 2;

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (lid == 0) {
		if (g_carry != NULL) {
			g_carry[gid] = l_data[LOCAL_DATA_SIZE - 1];
		}
		l_data[LOCAL_DATA_SIZE - 1] = 0;
	}

	d = LOCAL_DATA_SIZE / 2;
	for (work_group_bound = 1; work_group_bound != WORK_GROUP_SIZE * 2; work_group_bound *= 2) {
		barrier(CLK_LOCAL_MEM_FENCE);

		if (lid < work_group_bound) {
			ind_a = 2*d*lid + d - 1;
			ind_b = 2*d*lid + 2*d - 1;
			tmp = l_data[ind_b];
			l_data[ind_b] = l_data[ind_a] + tmp;
			l_data[ind_a] = tmp;
		}

		d = d / 2;
	}

	g_odata[2*gid] = l_data[2*lid];
	g_odata[2*gid + 1] = l_data[2*lid + 1];
}
