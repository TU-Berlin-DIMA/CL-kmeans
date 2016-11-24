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
#include <boost/compute/core.hpp>
#include <boost/compute/container/vector.hpp>

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

template <typename PointT, typename LabelT, typename MassT,
         bool ColMajor>
class ClusteringBenchmark {
public:
    template <typename T>
    using Vector = boost::compute::vector<T>;
    template <typename T>
    using VectorPtr = std::shared_ptr<boost::compute::vector<T>>;

    using ClusteringFunction = std::function<
        void(
            uint32_t,
            cle::Matrix<PointT, std::allocator<PointT>, size_t, ColMajor> const&,
            cle::Matrix<PointT, std::allocator<PointT>, size_t, ColMajor>&,
            std::vector<MassT, std::allocator<LabelT>>&,
            std::vector<LabelT, std::allocator<MassT>>&,
            Measurement::Measurement&
            )>;

    using ClClusteringFunction = std::function<
        void(
                size_t,
                size_t,
                std::shared_ptr<const std::vector<PointT>>,
                VectorPtr<PointT>,
                VectorPtr<MassT>,
                VectorPtr<LabelT>)
        >;

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
    ClusteringBenchmarkStats run(ClClusteringFunction f, boost::compute::command_queue q);
    void setVerificationReference(std::vector<LabelT>&& reference_labels);
    int setVerificationReference(ClusteringFunction reference);
    uint64_t verify(ClusteringFunction f);
    uint64_t verify(ClClusteringFunction f, boost::compute::command_queue q);
    void print_labels();

private:
    const uint32_t num_runs_;
    const size_t num_points_;
    size_t num_clusters_;
    const uint32_t max_iterations_;
    cle::Matrix<PointT, std::allocator<PointT>, size_t, ColMajor> const points_;
    cle::Matrix<PointT, std::allocator<PointT>, size_t, ColMajor> centroids_;
    std::vector<MassT> cluster_mass_;
    std::vector<LabelT> labels_;
    std::vector<LabelT> reference_labels_;
    InitCentroidsFunction init_centroids_;
};

}

extern template class cle::ClusteringBenchmark<float, uint32_t, uint32_t, true>;
extern template class cle::ClusteringBenchmark<double, uint64_t, uint64_t, true>;

#endif /* CLUSTERING_BENCHMARK_HPP */
