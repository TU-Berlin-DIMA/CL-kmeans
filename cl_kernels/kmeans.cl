double gaussian_distance(p_x, p_y, q_x, q_y) {
    double t_x = q_x - p_x;
    double t_y = q_y - p_y;

    return t_x * t_x + t_y * t_y;
}


__kernel
void kmeans(
            __global double *const g_points_x,
            __global double *const g_points_y,
            __global double *const g_centroids_x,
            __global double *const g_centroids_y,
            __global ulong *const g_cluster_assignment,
            __local double *const l_centroids_x,
            __local double *const l_centroids_y,
            __local double *const l_old_centroids_x,
            __local double *const l_old_centroids_y,
            __local ulong *const l_cluster_size,
            const ulong NUM_POINTS,
            const ulong NUM_CLUSTERS
       ) {

	const size_t gid = get_global_id(0);
	const size_t lid = get_local_id(0);
	const size_t WORK_GROUP_SIZE = get_local_size(0);
	const size_t WORK_ITEM_SIZE = NUM_POINTS / get_global_size(0);
	const ulong LOCAL_NUM_POINTS = NUM_POINTS / get_num_groups(0);

    ulong min_c;
    double min_dist;

    // Assume centroids fit into local memory
    // Copy centroids to local memory
    for (uint i = 0; i < NUM_CLUSTERS; i += WORK_GROUP_SIZE) {
        if (i + lid < NUM_CLUSTERS) {
            l_centroids_x[i + lid] = g_centroids_x[i + lid];
        }
    }


    for (ulong r = 0; r < NUM_POINTS; r += get_global_size(0)) {

        // Currently processing point number
        ulong p = gid + r;

        // Phase 1
        if (gid < NUM_POINTS) {
            double point_x = g_points_x[p];
            double point_y = g_points_y[p];

            min_c = -1;
            min_dist = -1;

            for (uint c = 0; c < NUM_CLUSTERS; ++c) {
                dist = gaussian_distance(point_x, point_y, l_centroids_x[c], l_centroids_y[c]);
                if (dist < min_dist) {
                    min_dist = dist;
                    min_c = c;
                }
            }

            g_cluster_assignment[p] = min_c;
            // TODO: accumulate min_dists
        }
    }

    // Phase 2

    l_cluster_size[lid] = 0;

    // TODO atomic increment?
    l_cluster_size[min_c] += 1;
    l_centroids_x[min_c] += point_x;
    l_centroids_y[min_c] += point_y;

    // TODO sync global
    // TODO centroid = centroid / cluster_size
}
