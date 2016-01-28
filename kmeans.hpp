#ifndef KMEANS_HPP
#define KMEANS_HPP

#include "kmeans_cl_api.hpp"

#include <vector>
#include <cstdint>

#ifdef MAC
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

namespace cle {

using KmeansStats = struct { uint32_t iterations; };

class KmeansInitializer {
public:
    static void random(
            std::vector<double> const& points_x,
            std::vector<double> const& points_y,
            std::vector<double>& centroids_x,
            std::vector<double>& centroids_y);

    static void first_x(
            std::vector<double> const &points_x,
            std::vector<double> const& points_y,
            std::vector<double>& centroids_x,
            std::vector<double>& centroids_y);
};

template <typename FP, typename INT>
class KmeansNaive {
public:
    int initialize();
    int finalize();

    void operator() (uint32_t const max_iterations,
                      std::vector<FP> const& points_x,
                      std::vector<FP> const& points_y,
                      std::vector<FP>& centroids_x,
                      std::vector<FP>& centroids_y,
                      std::vector<INT>& cluster_size,
                      std::vector<INT>& memberships,
                      KmeansStats& stats);
private:
    double gaussian_distance(FP a_x, FP a_y, FP b_x, FP b_y);

};

class KmeansGPUAssisted {
public:
    KmeansGPUAssisted(
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
            std::vector<uint64_t>& memberships,
            KmeansStats& stats
            );

private:
    cle::Kmeans_With_Host_Kernel kmeans_kernel_;
    cl::Context context_;
    cl::CommandQueue queue_;

    size_t max_work_group_size_;
    std::vector<size_t> max_work_item_sizes_;
};

using KmeansNaive32 = KmeansNaive<float, uint32_t>;
using KmeansNaive64 = KmeansNaive<double, uint64_t>;

}

extern template class cle::KmeansNaive<float, uint32_t>;
extern template class cle::KmeansNaive<double, uint64_t>;

#endif /* KMEANS_HPP */
