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

#include <boost/align/aligned_allocator.hpp>

namespace cle {

using KmeansStats = struct { uint32_t iterations; };
using AlignedAllocatorFP32 =
    boost::alignment::aligned_allocator<float, 256>;
using AlignedAllocatorINT32 =
    boost::alignment::aligned_allocator<uint32_t, 256>;
using AlignedAllocatorFP64 =
    boost::alignment::aligned_allocator<double, 256>;
using AlignedAllocatorINT64 =
    boost::alignment::aligned_allocator<uint64_t, 256>;

template <typename FP, typename Alloc>
class KmeansInitializer {
public:
    static void forgy(
            std::vector<FP, Alloc> const& points_x,
            std::vector<FP, Alloc> const& points_y,
            std::vector<FP, Alloc>& centroids_x,
            std::vector<FP, Alloc>& centroids_y);

    static void first_x(
            std::vector<FP, Alloc> const &points_x,
            std::vector<FP, Alloc> const& points_y,
            std::vector<FP, Alloc>& centroids_x,
            std::vector<FP, Alloc>& centroids_y);
};

template <typename FP, typename INT, typename AllocFP, typename AllocINT>
class KmeansNaive {
public:
    int initialize();
    int finalize();

    void operator() (uint32_t const max_iterations,
                      std::vector<FP, AllocFP> const& points_x,
                      std::vector<FP, AllocFP> const& points_y,
                      std::vector<FP, AllocFP>& centroids_x,
                      std::vector<FP, AllocFP>& centroids_y,
                      std::vector<INT, AllocINT>& cluster_size,
                      std::vector<INT, AllocINT>& memberships,
                      KmeansStats& stats);
private:
    FP gaussian_distance(FP a_x, FP a_y, FP b_x, FP b_y);

};

class KmeansSIMD32 {
public:
    int initialize();
    int finalize();

    void operator() (
            uint32_t const max_iterations,
            std::vector<float, AlignedAllocatorFP32> const& points_x,
            std::vector<float, AlignedAllocatorFP32> const& points_y,
            std::vector<float, AlignedAllocatorFP32>& centroids_x,
            std::vector<float, AlignedAllocatorFP32>& centroids_y,
            std::vector<uint32_t, AlignedAllocatorINT32>& cluster_size,
            std::vector<uint32_t, AlignedAllocatorINT32>& memberships,
            KmeansStats& stats);
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

using KmeansInitializer32 =
    KmeansInitializer<float, std::allocator<float>>;
using KmeansInitializer64 =
    KmeansInitializer<double, std::allocator<double>>;
using KmeansInitializer32Aligned =
    KmeansInitializer<float, AlignedAllocatorFP32>;
using KmeansInitializer64Aligned =
    KmeansInitializer<double, AlignedAllocatorFP64>;

using KmeansNaive32 =
    KmeansNaive<float, uint32_t, std::allocator<float>, std::allocator<uint32_t>>;
using KmeansNaive64 =
    KmeansNaive<double, uint64_t, std::allocator<double>, std::allocator<uint64_t>>;
using KmeansNaive32Aligned =
    KmeansNaive<float, uint32_t, AlignedAllocatorFP32, AlignedAllocatorINT32>;
using KmeansNaive64Aligned =
    KmeansNaive<double, uint64_t, AlignedAllocatorFP64, AlignedAllocatorINT64>;

}

extern template class cle::KmeansInitializer<float, std::allocator<float>>;
extern template class cle::KmeansInitializer<double, std::allocator<double>>;
extern template class cle::KmeansInitializer<float, cle::AlignedAllocatorFP32>;
extern template class cle::KmeansInitializer<double, cle::AlignedAllocatorFP64>;

extern template class cle::KmeansNaive<float, uint32_t, std::allocator<float>, std::allocator<uint32_t>>;
extern template class cle::KmeansNaive<double, uint64_t, std::allocator<double>, std::allocator<uint64_t>>;
extern template class cle::KmeansNaive<float, uint32_t, cle::AlignedAllocatorFP32, cle::AlignedAllocatorINT32>;
extern template class cle::KmeansNaive<double, uint64_t, cle::AlignedAllocatorFP64, cle::AlignedAllocatorINT64>;

#endif /* KMEANS_HPP */
