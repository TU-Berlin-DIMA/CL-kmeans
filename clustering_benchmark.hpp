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
#include "matrix.hpp"
#include "measurement/measurement.hpp"

#include <vector>
#include <memory>
#include <functional>
#include <cstdint>
#include <type_traits>

namespace Clustering {

class ClusteringBenchmarkStats {
public:
    ClusteringBenchmarkStats(const uint32_t num_runs);

    void set_dimensions(
            uint64_t num_features,
            uint64_t num_points,
            uint64_t num_clusters
            );

    void print_times();
    void to_csv(char const* csv_file, char const* input_file);

    std::vector<uint64_t> microseconds;
    std::vector<std::shared_ptr<Measurement::Measurement>> measurements;

private:
    uint32_t num_runs_;
    uint64_t num_features_, num_points_, num_clusters_;

    static char const *const parameters_suffix_;
    static char const *const iterated_measurements_suffix_;
    static char const *const onetime_measurements_suffix_;

    static uint32_t const max_hostname_length_;
    static uint32_t const max_datetime_length_;
    static char const *const timestamp_format_;
};

template <typename PointT, typename LabelT, typename MassT,
         bool ColMajor>
class ClusteringBenchmark {
public:
    using ClusteringFunction = std::function<
        std::shared_ptr<Measurement::Measurement>(
            uint32_t,
            cle::Matrix<PointT, std::allocator<PointT>, size_t, ColMajor> const&,
            cle::Matrix<PointT, std::allocator<PointT>, size_t, ColMajor>&,
            std::vector<MassT, std::allocator<LabelT>>&,
            std::vector<LabelT, std::allocator<MassT>>&
            )>;

    using ClClusteringFunction = std::function<
        std::shared_ptr<Measurement::Measurement>(
                size_t,
                size_t,
                std::shared_ptr<const std::vector<PointT>>,
                std::shared_ptr<std::vector<PointT>>,
                std::shared_ptr<std::vector<MassT>>,
                std::shared_ptr<std::vector<LabelT>>
                )>;

    using InitCentroidsFunction = std::function<
        void(
            cle::Matrix<PointT, std::allocator<PointT>, size_t, ColMajor> const&,
            cle::Matrix<PointT, std::allocator<PointT>, size_t, ColMajor>&
            )>;

    ClusteringBenchmark(
            const uint32_t num_runs,
            const size_t num_points,
            const size_t max_iterations,
            cle::Matrix<PointT, std::allocator<PointT>, size_t, ColMajor>&& points
            );

    ClusteringBenchmark(
            const uint32_t,
            const size_t,
            const uint32_t,
            cle::Matrix<PointT, std::allocator<PointT>, size_t, ColMajor>&
            ) = delete;

    int initialize(
            const size_t num_clusters,
            const size_t num_features,
            InitCentroidsFunction init_centroids
            );
    int finalize();

    ClusteringBenchmarkStats run(ClusteringFunction f);
    ClusteringBenchmarkStats run(ClClusteringFunction f);
    void setVerificationReference(std::vector<LabelT>&& reference_labels);
    int setVerificationReference(ClusteringFunction reference);
    uint64_t verify(ClusteringFunction f);
    uint64_t verify(ClClusteringFunction f);
    double mse();
    void print_labels();
    void print_result();

private:
    const uint32_t num_runs_;
    const size_t num_points_;
    size_t num_clusters_;
    const uint32_t max_iterations_;
    cle::Matrix<PointT, std::allocator<PointT>, size_t, ColMajor> const points_;
    cle::Matrix<PointT, std::allocator<PointT>, size_t, ColMajor> centroids_;
    cle::Matrix<PointT, std::allocator<PointT>, size_t, ColMajor> reference_centroids_;
    std::vector<MassT> cluster_mass_;
    std::vector<MassT> reference_cluster_mass_;
    std::vector<LabelT> labels_;
    std::vector<LabelT> reference_labels_;
    InitCentroidsFunction init_centroids_;
};

}

extern template class Clustering::ClusteringBenchmark<float, uint32_t, uint32_t, true>;
extern template class Clustering::ClusteringBenchmark<double, uint64_t, uint64_t, true>;

#endif /* CLUSTERING_BENCHMARK_HPP */
