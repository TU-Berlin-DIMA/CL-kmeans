/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 * 
 * 
 * Copyright (c) 2016, Lutz, Clemens <lutzcle@cml.li>
 */

#ifndef CLUSTERING_BENCHMARK_HPP
#define CLUSTERING_BENCHMARK_HPP

#include "timer.hpp"
#include "kmeans.hpp"
#include "matrix.hpp"
#include "measurement/measurement.hpp"

#include <vector>
#include <memory>
#include <functional>
#include <cstdint>
#include <type_traits>
#include <boost/compute/container/vector.hpp>
#include <boost/compute/container/mapped_view.hpp>

namespace cle {

class ClusteringBenchmarkStats {
public:
    ClusteringBenchmarkStats(const uint32_t num_runs);

    void set_dimensions(
            uint64_t num_features,
            uint64_t num_points,
            uint64_t num_clusters
            );

    template <typename FP, typename INT>
    void set_types() {
        is_uint32 = std::is_same<uint32_t, INT>::value;
        is_float32 = std::is_same<float, FP>::value;
    }

    void print_times();
    void to_csv(char const* csv_file, char const* input_file);

    std::vector<uint64_t> microseconds;
    std::vector<Measurement::Measurement> measurements;

private:
    uint32_t num_runs_;
    uint64_t num_features_, num_points_, num_clusters_;
    bool is_uint32, is_float32;

    static char const *const parameters_suffix_;
    static char const *const iterated_measurements_suffix_;
    static char const *const onetime_measurements_suffix_;

    static uint32_t const max_hostname_length_;
    static uint32_t const max_datetime_length_;
    static char const *const timestamp_format_;
};

template <typename FP, typename INT, typename AllocFP, typename AllocINT,
         bool COL_MAJOR>
class ClusteringBenchmark {
public:
    using PointT = FP;
    using LabelT = INT;
    using MassT = INT;

    template <typename T>
    using Vector = boost::compute::vector<T>;
    template <typename T>
    using VectorPtr = std::shared_ptr<boost::compute::vector<T>>;
    template <typename T>
    using MappedView = boost::compute::mapped_view<T>;

    using ClusteringFunction = std::function<
        void(
            uint32_t,
            cle::Matrix<FP, AllocFP, INT, COL_MAJOR> const&,
            cle::Matrix<FP, AllocFP, INT, COL_MAJOR>&,
            std::vector<INT, AllocINT>&,
            std::vector<INT, AllocINT>&,
            Measurement::Measurement&
            )>;

    using ClClusteringFunction = std::function<
        void(
                size_t,
                size_t,
                MappedView<PointT>,
                VectorPtr<PointT>,
                VectorPtr<MassT>,
                VectorPtr<LabelT>)
        >;

    using InitCentroidsFunction = std::function<
        void(
            cle::Matrix<FP, AllocFP, INT, COL_MAJOR> const&,
            cle::Matrix<FP, AllocFP, INT, COL_MAJOR>&
            )>;

    ClusteringBenchmark(
            const uint32_t num_runs,
            const INT num_points,
            const uint32_t max_iterations,
            cle::Matrix<FP, AllocFP, INT, COL_MAJOR>&& points
            );

    ClusteringBenchmark(
            const uint32_t,
            const INT,
            const uint32_t,
            cle::Matrix<FP, AllocFP, INT, COL_MAJOR>&
            ) = delete;

    int initialize(
            const INT num_clusters,
            const INT num_features,
            InitCentroidsFunction init_centroids
            );
    int finalize();

    ClusteringBenchmarkStats run(ClusteringFunction f);
    ClusteringBenchmarkStats run(ClClusteringFunction f);
    void setVerificationReference(std::vector<INT, AllocINT>&& reference_labels);
    int setVerificationReference(ClusteringFunction reference);
    uint64_t verify(ClusteringFunction f);
    uint64_t verify(ClClusteringFunction f);
    void print_labels();

private:
    const uint32_t num_runs_;
    const INT num_points_;
    INT num_clusters_;
    const uint32_t max_iterations_;
    cle::Matrix<FP, AllocFP, INT, COL_MAJOR> const points_;
    cle::Matrix<FP, AllocFP, INT, COL_MAJOR> centroids_;
    std::vector<INT, AllocINT> cluster_mass_;
    std::vector<INT, AllocINT> labels_;
    std::vector<INT, AllocINT> reference_labels_;
    InitCentroidsFunction init_centroids_;
};

using ClusteringBenchmark32 = ClusteringBenchmark<float, uint32_t, std::allocator<float>, std::allocator<uint32_t>, true>;
using ClusteringBenchmark64 = ClusteringBenchmark<double, uint64_t, std::allocator<double>, std::allocator<uint64_t>, true>;
#ifdef USE_ALIGNED_ALLOCATOR
using ClusteringBenchmark32Aligned = ClusteringBenchmark<float, uint32_t, AlignedAllocatorFP32, AlignedAllocatorINT32, true>;
using ClusteringBenchmark64Aligned = ClusteringBenchmark<double, uint64_t, AlignedAllocatorFP64, AlignedAllocatorINT64, true>;
#endif

}

extern template class cle::ClusteringBenchmark<float, uint32_t, std::allocator<float>, std::allocator<uint32_t>, true>;
extern template class cle::ClusteringBenchmark<double, uint64_t, std::allocator<double>, std::allocator<uint64_t>, true>;
#ifdef USE_ALIGNED_ALLOCATOR
extern template class cle::ClusteringBenchmark<float, uint32_t, cle::AlignedAllocatorFP32, cle::AlignedAllocatorINT32, true>;
extern template class cle::ClusteringBenchmark<double, uint64_t, cle::AlignedAllocatorFP64, cle::AlignedAllocatorINT64, true>;
#endif

#endif /* CLUSTERING_BENCHMARK_HPP */
