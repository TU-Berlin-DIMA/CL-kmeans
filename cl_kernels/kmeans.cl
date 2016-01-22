#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable

double gaussian_distance(p_x, p_y, q_x, q_y) {
    double t_x = q_x - p_x;
    double t_y = q_y - p_y;

    return t_x * t_x + t_y * t_y;
}


__kernel
void kmeans(
            __global char *g_did_changes,
            __global double *const g_points_x,
            __global double *const g_points_y,
            __global double *const g_point_distance,
            __global double *const g_centroids_x,
            __global double *const g_centroids_y,
            __global ulong *const g_cluster_size,
            __global ulong *const g_cluster_assignment,
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
    ulong min_c;
    double min_dist;

    // Assume centroids fit into local memory

    // Read to local memory
    for (uint i = 0; i < NUM_CLUSTERS; i += WORK_GROUP_SIZE) {
        if (i + lid < NUM_CLUSTERS) {
            l_old_centroids_x[i + lid] = g_centroids_x[i + lid];
            l_old_centroids_y[i + lid] = g_centroids_y[i + lid];
            l_centroids_x[i + lid] = 0;
            l_centroids_y[i + lid] = 0;
        }
    }


    for (ulong r = 0; r < NUM_POINTS; r += get_global_size(0)) {

        // Currently processing point number
        ulong p = gid + r;

        if (gid < NUM_POINTS) {
            // Phase 1
            double point_x = g_points_x[p];
            double point_y = g_points_y[p];
            ulong cluster_assignment = g_cluster_assignment[p];

            min_c = ULONG_MAX;
            min_dist = DBL_MAX;

            for (uint c = 0; c < NUM_CLUSTERS; ++c) {
                dist = gaussian_distance(point_x, point_y, l_old_centroids_x[c], l_old_centroids_y[c]);
                if (dist < min_dist) {
                    min_dist = dist;
                    min_c = c;
                }
            }

            if (min_c != cluster_assignment) {
                did_changes = true;
            }

            g_cluster_assignment[p] = min_c;
            g_point_distance[p] = min_dist;

            // Phase 2
            // Note: This is not coalesced
            // http://stackoverflow.com/questions/22367238/cuda-atomic-operation-performance-in-different-scenarios
            // Note: Atomic local add is software implemented
            atom_inc(&g_cluster_size[min_c]);
            atom_add(&g_centroids_x[min_c], point_x);
            atom_add(&g_centroids_y[min_c], point_y);
        }
    }


    // Write back to global memory
    g_did_changes[gid] = did_changes;


    // for (uint i = 0; i < NUM_CLUSTERS; i += WORK_GROUP_SIZE) {
    //     if (i + lid < NUM_CLUSTERS) {
    //         g_centroids_x[i + lid] = l_centroids_x[i + lid];
    //         g_centroids_y[i + lid] = l_centroids_y[i + lid];
    //         g_cluster_size[i + lid] = l_cluster_size[i + lid];
    //     }
    // }


    // TODO centroid = centroid / cluster_size
}
