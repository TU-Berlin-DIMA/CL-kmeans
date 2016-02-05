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

#include <vector>
#include <iostream>
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

template <typename FP, typename INT, typename AllocFP, typename AllocINT>
class ClusteringBenchmark {
public:
    using ClusteringFunction = std::function<
        void(
            uint32_t,
            std::vector<FP, AllocFP> const&,
            std::vector<FP, AllocFP> const&,
            std::vector<FP, AllocFP>&,
            std::vector<FP, AllocFP>&,
            std::vector<INT, AllocINT>&,
            std::vector<INT, AllocINT>&,
            cle::KmeansStats&
            )>;

    using InitCentroidsFunction = std::function<
        void(
            std::vector<FP, AllocFP> const&,
            std::vector<FP, AllocFP> const&,
            std::vector<FP, AllocFP>&,
            std::vector<FP, AllocFP>&
            )>;

    ClusteringBenchmark(
            const uint32_t num_runs,
            const INT num_points,
            const uint32_t max_iterations,
            std::vector<FP, AllocFP>&& points_x,
            std::vector<FP, AllocFP>&& points_y
            );

    ClusteringBenchmark(
            const uint32_t,
            const INT,
            const uint32_t,
            std::vector<FP, AllocFP>&,
            std::vector<FP, AllocFP>&
            ) = delete;

    int initialize(
            const INT num_clusters,
            InitCentroidsFunction init_centroids
            );
    int finalize();

    ClusteringBenchmarkStats run(ClusteringFunction f);
    int setVerificationReference(ClusteringFunction reference);
    int verify(ClusteringFunction f);

private:
    const uint32_t num_runs_;
    const INT num_points_;
    INT num_clusters_;
    const uint32_t max_iterations_;
    std::vector<FP, AllocFP> const points_x_;
    std::vector<FP, AllocFP> const points_y_;
    std::vector<FP, AllocFP> centroids_x_;
    std::vector<FP, AllocFP> centroids_y_;
    std::vector<INT, AllocINT> cluster_size_;
    std::vector<INT, AllocINT> memberships_;
    std::vector<INT, AllocINT> reference_memberships_;
    InitCentroidsFunction init_centroids_;
};

using ClusteringBenchmark32 = ClusteringBenchmark<float, uint32_t, std::allocator<float>, std::allocator<uint32_t>>;
using ClusteringBenchmark64 = ClusteringBenchmark<double, uint64_t, std::allocator<double>, std::allocator<uint64_t>>;
using ClusteringBenchmark32Aligned = ClusteringBenchmark<float, uint32_t, AlignedAllocatorFP32, AlignedAllocatorINT32>;
using ClusteringBenchmark64Aligned = ClusteringBenchmark<double, uint64_t, AlignedAllocatorFP64, AlignedAllocatorINT64>;

}

extern template class cle::ClusteringBenchmark<float, uint32_t, std::allocator<float>, std::allocator<uint32_t>>;
extern template class cle::ClusteringBenchmark<double, uint64_t, std::allocator<double>, std::allocator<uint64_t>>;
extern template class cle::ClusteringBenchmark<float, uint32_t, cle::AlignedAllocatorFP32, cle::AlignedAllocatorINT32>;
extern template class cle::ClusteringBenchmark<double, uint64_t, cle::AlignedAllocatorFP64, cle::AlignedAllocatorINT64>;

#endif /* CLUSTERING_BENCHMARK_HPP */
