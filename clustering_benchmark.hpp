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

#include <vector>
#include <memory>
#include <functional>
#include <cstdint>

namespace cle {

class ClusteringBenchmarkStats {
public:
    ClusteringBenchmarkStats(const uint32_t num_runs);

    void print_times();

    std::vector<uint64_t> microseconds;
    std::vector<cle::KmeansStats> kmeans_stats;

private:
    const uint32_t num_runs_;
};

template <typename FP, typename INT, typename AllocFP, typename AllocINT,
         bool COL_MAJOR>
class ClusteringBenchmark {
public:
    using ClusteringFunction = std::function<
        void(
            uint32_t,
            cle::Matrix<FP, AllocFP, INT, COL_MAJOR> const&,
            cle::Matrix<FP, AllocFP, INT, COL_MAJOR>&,
            std::vector<INT, AllocINT>&,
            std::vector<INT, AllocINT>&,
            cle::KmeansStats&
            )>;

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
    int setVerificationReference(ClusteringFunction reference);
    int verify(ClusteringFunction f);
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
using ClusteringBenchmark32Aligned = ClusteringBenchmark<float, uint32_t, AlignedAllocatorFP32, AlignedAllocatorINT32, true>;
using ClusteringBenchmark64Aligned = ClusteringBenchmark<double, uint64_t, AlignedAllocatorFP64, AlignedAllocatorINT64, true>;

}

extern template class cle::ClusteringBenchmark<float, uint32_t, std::allocator<float>, std::allocator<uint32_t>, true>;
extern template class cle::ClusteringBenchmark<double, uint64_t, std::allocator<double>, std::allocator<uint64_t>, true>;
extern template class cle::ClusteringBenchmark<float, uint32_t, cle::AlignedAllocatorFP32, cle::AlignedAllocatorINT32, true>;
extern template class cle::ClusteringBenchmark<double, uint64_t, cle::AlignedAllocatorFP64, cle::AlignedAllocatorINT64, true>;

#endif /* CLUSTERING_BENCHMARK_HPP */
