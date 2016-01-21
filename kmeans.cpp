#include "kmeans.hpp"

#include <cstdint>

#include "kmeans_cl_api.hpp"

int cle::kmeans_gpu(
        uint32_t const max_iterations,
        std::vector<double> const& points_x,
        std::vector<double> const& points_y,
        std::vector<double>& centroids_x,
        std::vector<double>& centroids_y,
        std::vector<uint64_t>& cluster_size,
        std::vector<double>& cluster_assignment
        ) {

    uint32_t iterations;
    bool did_changes;

    // copy points (x,y) host -> device
    // copy centroids (x,y) host -> device

    iterations = 0;
    did_changes = true;
    while (did_changes && iterations < max_iterations) {

        // set did_changes to false on device
        // execute kernel
        // copy did_changes device -> host
        // aggregate did_changes
        // flip old_centroids <-> centroids

        ++iterations;
    }

    // copy centroids (x,y) device -> host
    // copy cluster sizes device -> host
    // copy point assignments device -> host

    return 1;
}


