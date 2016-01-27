#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable

double gaussian_distance(double p_x, double p_y, double q_x, double q_y) {
    double t_x = q_x - p_x;
    double t_y = q_y - p_y;

    return t_x * t_x + t_y * t_y;
}


__kernel
void kmeans_with_host(
            __global char *g_did_changes,
            __global double const *const g_points_x,
            __global double const *const g_points_y,
            __global double const *const g_centroids_x,
            __global double const *const g_centroids_y,
            __global ulong *const g_memberships,
            __local double *const l_centroids_x,
            __local double *const l_centroids_y,
            __local double *const l_old_centroids_x,
            __local double *const l_old_centroids_y,
            const ulong NUM_POINTS,
            const ulong NUM_CLUSTERS
       ) {

	const size_t gid = get_global_id(0);
	const size_t lid = get_local_id(0);
	const size_t WORK_GROUP_SIZE = get_local_size(0);
	const size_t WORK_ITEM_SIZE = NUM_POINTS / get_global_size(0);
	const ulong LOCAL_NUM_POINTS = NUM_POINTS / get_num_groups(0);

    bool did_changes = false;

    // Assume centroids fit into local memory

    // Read to local memory
    for (ulong i = 0; i < NUM_CLUSTERS; i += WORK_GROUP_SIZE) {
        if (i + lid < NUM_CLUSTERS) {
            l_old_centroids_x[i + lid] = g_centroids_x[i + lid];
            l_old_centroids_y[i + lid] = g_centroids_y[i + lid];
        }
    }
	barrier(CLK_LOCAL_MEM_FENCE);

    for (ulong r = 0; r < NUM_POINTS; r += get_global_size(0)) {

        // Currently processing point number
        ulong p = gid + r;

        if (p < NUM_POINTS) {
            // Phase 1
            double point_x = g_points_x[p];
            double point_y = g_points_y[p];
            ulong membership = g_memberships[p];

            ulong min_c;
            double min_dist = DBL_MAX;

            for (ulong c = 0; c < NUM_CLUSTERS; ++c) {
                double dist = gaussian_distance(point_x, point_y, l_old_centroids_x[c], l_old_centroids_y[c]);
                if (dist < min_dist) {
                    min_dist = dist;
                    min_c = c;
                }
            }

            if (min_c != membership) {
                did_changes = true;
            }

            g_memberships[p] = min_c;
        }
    }


    // Write back to global memory
    if (did_changes == true) {
        // g_did_changes[gid] = did_changes;
        *g_did_changes = true;
    }
}
