/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright (c) 2016, Lutz, Clemens <lutzcle@cml.li>
 */

#include "clustering_benchmark.hpp"

#include "measurement/measurement.hpp"

#include <iostream>
#include <cstdint>
#include <chrono>
#include <vector>
#include <string>
#include <boost/filesystem/path.hpp>
#include <boost/compute/container/vector.hpp>
#include <unistd.h>

#include <Version.h>

uint32_t const max_hostname_length = 30;

cle::ClusteringBenchmarkStats::ClusteringBenchmarkStats(const uint32_t num_runs)
    :
        microseconds(num_runs),
        measurements(num_runs),
        num_runs_(num_runs)
{}

void cle::ClusteringBenchmarkStats::set_dimensions(
        uint64_t num_features,
        uint64_t num_points,
        uint64_t num_clusters
        ) {

    num_features_ = num_features;
    num_points_ = num_points;
    num_clusters_ = num_clusters;
}

void cle::ClusteringBenchmarkStats::print_times() {
    std::cout << num_runs_ << " runs, in µs: [";
    for (uint32_t r = 0; r < microseconds.size(); ++r) {
        std::cout << microseconds[r];
        if (r != microseconds.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;
}

void cle::ClusteringBenchmarkStats::to_csv(
        char const* csv_file,
        char const* input_file
        ) {

    assert(microseconds.size() == measurements.size());

    char hostname[max_hostname_length];
    gethostname(hostname, max_hostname_length);

    boost::filesystem::path input_file_path(input_file);
    boost::filesystem::path input_file_name = input_file_path.filename();

    std::string int_type = is_uint32 ? "uint32_t" : "uint64_t";
    std::string float_type = is_float32 ? "float" : "double";

    for (Measurement::Measurement& m : measurements) {
        m.set_parameter(
                Measurement::ParameterType::Version,
                GIT_REVISION
                );
        m.set_parameter(
                Measurement::ParameterType::Filename,
                input_file_name.c_str()
                );
        m.set_parameter(
                Measurement::ParameterType::Hostname,
                hostname
                );
        m.set_parameter(
                Measurement::ParameterType::NumFeatures,
                std::to_string(num_features_)
                );
        m.set_parameter(
                Measurement::ParameterType::NumPoints,
                std::to_string(num_points_)
                );
        m.set_parameter(
                Measurement::ParameterType::NumClusters,
                std::to_string(num_clusters_)
                );
        m.set_parameter(
                Measurement::ParameterType::IntType,
                int_type
                );
        m.set_parameter(
                Measurement::ParameterType::FloatType,
                float_type
                );

        m.write_csv(csv_file);
    }

}

template <typename PointT, typename LabelT, typename MassT, bool ColMajor>
cle::ClusteringBenchmark<PointT, LabelT, MassT, ColMajor>::ClusteringBenchmark(
        const uint32_t num_runs,
        const size_t num_points,
        const size_t max_iterations,
        cle::Matrix<PointT, std::allocator<PointT>, size_t, ColMajor>&& points
        )
    :
        num_runs_(num_runs),
        num_points_(num_points),
        num_clusters_(0),
        max_iterations_(max_iterations),
        points_(std::move(points)),
        labels_(num_points)
{}

template <typename PointT, typename LabelT, typename MassT, bool ColMajor>
int cle::ClusteringBenchmark<PointT, LabelT, MassT, ColMajor>::initialize(
        const size_t num_clusters, const size_t num_features,
        InitCentroidsFunction init_centroids
        ) {

    num_clusters_ = num_clusters;
    init_centroids_ = init_centroids;

    centroids_.resize(num_clusters, num_features);
    cluster_mass_.resize(num_clusters);

    return 1;
}

template <typename PointT, typename LabelT, typename MassT, bool ColMajor>
int cle::ClusteringBenchmark<PointT, LabelT, MassT, ColMajor>::finalize() {
    return 1;
}

template <typename PointT, typename LabelT, typename MassT, bool ColMajor>
cle::ClusteringBenchmarkStats cle::ClusteringBenchmark<PointT, LabelT, MassT, ColMajor>::run(
        ClusteringFunction f) {

    cle::Timer timer;
    ClusteringBenchmarkStats bs(this->num_runs_);
    bs.set_dimensions(points_.cols(), points_.rows(), centroids_.rows());
    bs.set_types<PointT, MassT>();

    for (uint32_t r = 0; r < this->num_runs_; ++r) {
        init_centroids_(
                points_,
                centroids_
                );

        timer.start();
        f(
                max_iterations_,
                points_,
                centroids_,
                cluster_mass_,
                labels_,
                bs.measurements[r]
         );
        bs.microseconds[r] = timer.stop<std::chrono::microseconds>();
    }

    return bs;
}

template <typename PointT, typename LabelT, typename MassT, bool ColMajor>
cle::ClusteringBenchmarkStats cle::ClusteringBenchmark<PointT, LabelT, MassT, ColMajor>::run(
        ClClusteringFunction f,
        boost::compute::command_queue queue) {

    cle::Timer timer;
    ClusteringBenchmarkStats bs(this->num_runs_);
    bs.set_dimensions(points_.cols(), points_.rows(), centroids_.rows());
    bs.set_types<PointT, MassT>();

    // Dirty hack to avoid freeing object
    // when shared_ptr goes out of scope.
    // Should convert ClusteringBenchmark to use shared_ptr's.
    std::shared_ptr<const std::vector<PointT>> points(
            &this->points_.get_data(),
            [](const std::vector<PointT> *){}
            );
    VectorPtr<MassT> masses = std::make_shared<Vector<MassT>>(
            this->cluster_mass_,
            queue);
    VectorPtr<LabelT> labels = std::make_shared<Vector<LabelT>>(
            this->labels_,
            queue);

    for (uint32_t r = 0; r < this->num_runs_; ++r) {
        init_centroids_(
                points_,
                centroids_
                );

        std::shared_ptr<Measurement::Measurement> measurement(
                &bs.measurements[r],
                [](const Measurement::Measurement *){}
                );

        VectorPtr<PointT> centroids = std::make_shared<Vector<PointT>>(
                this->centroids_.get_data(),
                queue);

        timer.start();
        f(
                max_iterations_,
                points_.cols(),
                points,
                centroids,
                masses,
                labels,
                measurement
         );
        bs.microseconds[r] = timer.stop<std::chrono::microseconds>();
    }

    return bs;
}

template <typename PointT, typename LabelT, typename MassT, bool ColMajor>
void cle::ClusteringBenchmark<PointT, LabelT, MassT, ColMajor>::setVerificationReference(std::vector<LabelT>&& reference_labels) {

    reference_labels_ = std::move(reference_labels);
}

template <typename PointT, typename LabelT, typename MassT, bool ColMajor>
int cle::ClusteringBenchmark<PointT, LabelT, MassT, ColMajor>::setVerificationReference(
        ClusteringFunction ref) {

    Measurement::Measurement stats;

    reference_labels_.resize(num_points_);

    init_centroids_(
            points_,
            centroids_
            );

    ref(
            max_iterations_,
            points_,
            centroids_,
            cluster_mass_,
            reference_labels_,
            stats
       );

    return 1;
}

template <typename PointT, typename LabelT, typename MassT, bool ColMajor>
uint64_t cle::ClusteringBenchmark<PointT, LabelT, MassT, ColMajor>::verify(ClusteringFunction f) {

    Measurement::Measurement stats;

    init_centroids_(
            points_,
            centroids_
            );

    f(
            max_iterations_,
            points_,
            centroids_,
            cluster_mass_,
            labels_,
            stats
       );

    uint64_t counter = 0;
    for (size_t l = 0; l < labels_.size(); ++l) {
        if (reference_labels_[l] != labels_[l]) {
            ++counter;
        }
    }

    return counter;
}

template <typename PointT, typename LabelT, typename MassT, bool ColMajor>
uint64_t cle::ClusteringBenchmark<PointT, LabelT, MassT, ColMajor>::verify(
        ClClusteringFunction f,
        boost::compute::command_queue queue) {

    init_centroids_(
            points_,
            centroids_
            );

    std::shared_ptr<const std::vector<PointT>> points(
            &this->points_.get_data(),
            [](const std::vector<PointT> *){}
            );
    VectorPtr<PointT> centroids = std::make_shared<Vector<PointT>>(
            this->centroids_.get_data(),
            queue);
    VectorPtr<MassT> masses = std::make_shared<Vector<MassT>>(
            this->cluster_mass_,
            queue);
    VectorPtr<LabelT> labels = std::make_shared<Vector<LabelT>>(
            this->labels_,
            queue);

    f(
            max_iterations_,
            points_.cols(),
            points,
            centroids,
            masses,
            labels,
            nullptr
     );

    boost::compute::copy(
            labels->begin(),
            labels->end(),
            this->labels_.begin(),
            queue);

    uint64_t counter = 0;
    for (size_t l = 0; l < labels_.size(); ++l) {
        if (reference_labels_[l] != labels_[l]) {
            ++counter;
        }
    }

    return counter;
}

template <typename PointT, typename LabelT, typename MassT, bool ColMajor>
void cle::ClusteringBenchmark<PointT, LabelT, MassT, ColMajor>::print_labels() {

    std::cout << "Point Label" << std::endl;
    for (size_t i = 0; i < labels_.size(); ++i) {
        std::cout << i << " " << labels_[i] << std::endl;
    }
}

template class cle::ClusteringBenchmark<float, uint32_t, uint32_t, true>;
template class cle::ClusteringBenchmark<double, uint64_t, uint64_t, true>;
