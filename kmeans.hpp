#ifndef KMEANS_HPP
#define KMEANS_HPP

#include <vector>
#include <cstdint>

namespace cle {

    int kmeans_gpu(
            uint32_t const max_iterations,
            std::vector<double> const& points_x,
            std::vector<double> const& points_y,
            std::vector<double>& centroids_x,
            std::vector<double>& centroids_y,
            std::vector<uint64_t>& cluster_size,
            std::vector<double>& cluster_assignment
            );

}
#endif /* KMEANS_HPP */
