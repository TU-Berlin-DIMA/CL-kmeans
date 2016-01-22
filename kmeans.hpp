#ifndef KMEANS_HPP
#define KMEANS_HPP

#include <vector>
#include <cstdint>
#include <cstddef> // size_t

#ifdef MAC
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

#include "kmeans_cl_api.hpp"

namespace cle {
class Kmeans_GPU {
public:
    Kmeans_GPU(
            cl::Context const& context,
            cl::CommandQueue const& queue
            );

    int initialize();
    int finalize();

    int operator() (
            uint32_t const max_iterations,
            std::vector<double> const& points_x,
            std::vector<double> const& points_y,
            std::vector<double>& centroids_x,
            std::vector<double>& centroids_y,
            std::vector<uint64_t>& cluster_size,
            std::vector<uint64_t>& cluster_assignment
            );

private:
    cle::Kmeans_Kernel kmeans_kernel_;
    cl::Context context_;
    cl::CommandQueue queue_;

    size_t max_work_group_size_;
    std::vector<size_t> max_work_item_sizes_;
};
}
#endif /* KMEANS_HPP */
