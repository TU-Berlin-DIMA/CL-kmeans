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

template <typename FP, typename INT>
class ClusteringBenchmark {
public:
    using ClusteringFunction = std::function<
        void(
            uint32_t,
            std::vector<FP> const&,
            std::vector<FP> const&,
            std::vector<FP>&,
            std::vector<FP>&,
            std::vector<INT>&,
            std::vector<INT>&,
            cle::KmeansStats&
            )>;

    using InitCentroidsFunction = std::function<
        void(
            std::vector<FP> const&,
            std::vector<FP> const&,
            std::vector<FP>&,
            std::vector<FP>&
            )>;

    ClusteringBenchmark(
            const uint32_t num_runs,
            const INT num_points,
            const uint32_t max_iterations,
            std::vector<FP>&& points_x,
            std::vector<FP>&& points_y
            );

    ClusteringBenchmark(
            const uint32_t,
            const INT,
            const uint32_t,
            std::vector<FP>&,
            std::vector<FP>&
            ) = delete;

    int initialize(
            const INT num_clusters,
            InitCentroidsFunction init_centroids
            );
    int finalize();

    ClusteringBenchmarkStats run(ClusteringFunction f);

private:
    const uint32_t num_runs_;
    const INT num_points_;
    INT num_clusters_;
    const uint32_t max_iterations_;
    std::vector<FP> const points_x_;
    std::vector<FP> const points_y_;
    std::vector<FP> centroids_x_;
    std::vector<FP> centroids_y_;
    std::vector<INT> cluster_size_;
    std::vector<INT> memberships_;
    InitCentroidsFunction init_centroids_;
};

}

extern template class cle::ClusteringBenchmark<float, uint32_t>;
extern template class cle::ClusteringBenchmark<double, uint64_t>;

#endif /* CLUSTERING_BENCHMARK_HPP */
