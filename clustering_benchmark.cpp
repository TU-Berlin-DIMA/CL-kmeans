/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright (c) 2016-2018, Lutz, Clemens <lutzcle@cml.li>"
 */

#include "clustering_benchmark.hpp"

#include "measurement/measurement.hpp"

#include "utility.hpp"

#include <iostream>
#include <cstdint>
#include <chrono>
#include <vector>
#include <string>
#include <boost/filesystem/path.hpp>
#include <unistd.h>

#include <Version.h>

uint32_t const max_hostname_length = 30;

Clustering::ClusteringBenchmarkStats::ClusteringBenchmarkStats(const uint32_t num_runs)
    :
        num_runs_(num_runs)
{}

void Clustering::ClusteringBenchmarkStats::set_dimensions(
        uint64_t num_features,
        uint64_t num_points,
        uint64_t num_clusters
        ) {

    num_features_ = num_features;
    num_points_ = num_points;
    num_clusters_ = num_clusters;
}

void Clustering::ClusteringBenchmarkStats::print_times() {
    std::cout << num_runs_ << " runs, in µs: [";
    for (uint32_t r = 0; r < microseconds.size(); ++r) {
        std::cout << microseconds[r];
        if (r != microseconds.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;
}

void Clustering::ClusteringBenchmarkStats::to_csv(
        char const* csv_file,
        char const* input_file
        ) {

    assert(microseconds.size() == measurements.size());

    char hostname[max_hostname_length];
    gethostname(hostname, max_hostname_length);

    boost::filesystem::path input_file_path(input_file);
    boost::filesystem::path input_file_name = input_file_path.filename();

    for (auto& m : measurements) {
        m->set_parameter(
                "Version",
                GIT_REVISION
                );
        m->set_parameter(
                "Filename",
                input_file_name.c_str()
                );
        m->set_parameter(
                "Hostname",
                hostname
                );
        m->set_parameter(
                "NumFeatures",
                std::to_string(num_features_)
                );
        m->set_parameter(
                "NumPoints",
                std::to_string(num_points_)
                );
        m->set_parameter(
                "NumClusters",
                std::to_string(num_clusters_)
                );

        m->write_csv(csv_file);
    }

}

template <typename PointT, typename LabelT, typename MassT, bool ColMajor>
Clustering::ClusteringBenchmark<PointT, LabelT, MassT, ColMajor>::ClusteringBenchmark(
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
int Clustering::ClusteringBenchmark<PointT, LabelT, MassT, ColMajor>::initialize(
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
int Clustering::ClusteringBenchmark<PointT, LabelT, MassT, ColMajor>::finalize() {
    return 1;
}

template <typename PointT, typename LabelT, typename MassT, bool ColMajor>
Clustering::ClusteringBenchmarkStats Clustering::ClusteringBenchmark<PointT, LabelT, MassT, ColMajor>::run(
        ClusteringFunction f) {

    Timer::Timer timer;
    std::shared_ptr<Measurement::Measurement> measurement;
    ClusteringBenchmarkStats bs(this->num_runs_);
    bs.set_dimensions(points_.cols(), points_.rows(), centroids_.rows());

    for (uint32_t r = 0; r < this->num_runs_; ++r) {
        init_centroids_(
                points_,
                centroids_
                );

        timer.start();
        measurement = f(
                max_iterations_,
                points_,
                centroids_,
                cluster_mass_,
                labels_
                );
        bs.microseconds
            .push_back(timer.stop<std::chrono::microseconds>());
        measurement->set_run(r);
        bs.measurements.push_back(measurement);
    }

    return bs;
}

template <typename PointT, typename LabelT, typename MassT, bool ColMajor>
Clustering::ClusteringBenchmarkStats Clustering::ClusteringBenchmark<PointT, LabelT, MassT, ColMajor>::run(ClClusteringFunction f) {

    Timer::Timer timer;
    std::shared_ptr<Measurement::Measurement> measurement;
    ClusteringBenchmarkStats bs(this->num_runs_);
    bs.set_dimensions(points_.cols(), points_.rows(), centroids_.rows());

    // Dirty hack to avoid freeing object
    // when shared_ptr goes out of scope.
    // Should convert ClusteringBenchmark to use shared_ptr's.
    std::shared_ptr<const std::vector<PointT>> points(
            &this->points_.get_data(),
            [](const std::vector<PointT> *){}
            );
    std::shared_ptr<std::vector<PointT>> centroids(
            &this->centroids_.get_data(),
            [](std::vector<PointT> *){}
            );
    std::shared_ptr<std::vector<MassT>> masses(
            &this->cluster_mass_,
            [](std::vector<MassT> *){}
            );
    std::shared_ptr<std::vector<LabelT>> labels(
            &this->labels_,
            [](std::vector<MassT> *){}
            );

    for (uint32_t r = 0; r < this->num_runs_; ++r) {
        init_centroids_(
                points_,
                centroids_
                );

        timer.start();
        measurement = f(
                max_iterations_,
                points_.cols(),
                points,
                centroids,
                masses,
                labels
                );
        bs.microseconds
            .push_back(timer.stop<std::chrono::microseconds>());
        measurement->set_run(r);
        bs.measurements.push_back(measurement);
    }

    return bs;
}

template <typename PointT, typename LabelT, typename MassT, bool ColMajor>
void Clustering::ClusteringBenchmark<PointT, LabelT, MassT, ColMajor>::setVerificationReference(std::vector<LabelT>&& reference_labels) {

    reference_labels_ = std::move(reference_labels);
}

template <typename PointT, typename LabelT, typename MassT, bool ColMajor>
int Clustering::ClusteringBenchmark<PointT, LabelT, MassT, ColMajor>::setVerificationReference(
        ClusteringFunction ref) {

    reference_centroids_.resize(centroids_.rows(), centroids_.cols());
    reference_cluster_mass_.resize(num_clusters_);
    reference_labels_.resize(num_points_);

    init_centroids_(
            points_,
            reference_centroids_
            );

    ref(
            max_iterations_,
            points_,
            reference_centroids_,
            reference_cluster_mass_,
            reference_labels_
       );

    return 1;
}

template <typename PointT, typename LabelT, typename MassT, bool ColMajor>
uint64_t Clustering::ClusteringBenchmark<PointT, LabelT, MassT, ColMajor>::verify(ClusteringFunction f) {

    init_centroids_(
            points_,
            centroids_
            );

    f(
            max_iterations_,
            points_,
            centroids_,
            cluster_mass_,
            labels_
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
uint64_t Clustering::ClusteringBenchmark<PointT, LabelT, MassT, ColMajor>::verify(ClClusteringFunction f) {

    init_centroids_(
            points_,
            centroids_
            );

    std::shared_ptr<const std::vector<PointT>> points(
            &this->points_.get_data(),
            [](const std::vector<PointT> *){}
            );
    std::shared_ptr<std::vector<PointT>> centroids(
            &this->centroids_.get_data(),
            [](std::vector<PointT> *){}
            );
    std::shared_ptr<std::vector<MassT>> masses(
            &this->cluster_mass_,
            [](std::vector<MassT> *){}
            );
    std::shared_ptr<std::vector<LabelT>> labels(
            &this->labels_,
            [](std::vector<MassT> *){}
            );

    f(
            max_iterations_,
            points_.cols(),
            points,
            centroids,
            masses,
            labels
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
double Clustering::ClusteringBenchmark<PointT, LabelT, MassT, ColMajor>::mse() {
    return Clustering::Utility::mse(
            centroids_.begin(),
            centroids_.end(),
            reference_centroids_.begin()
            );
}



template <typename PointT, typename LabelT, typename MassT, bool ColMajor>
void Clustering::ClusteringBenchmark<PointT, LabelT, MassT, ColMajor>::print_labels() {

    std::cout << "Point Label" << std::endl;
    for (size_t i = 0; i < labels_.size(); ++i) {
        std::cout << i << " " << labels_[i] << std::endl;
    }
}

template <typename PointT, typename LabelT, typename MassT, bool ColMajor>
void Clustering::ClusteringBenchmark<PointT, LabelT, MassT, ColMajor>::print_result() {

    std::cout << "Centroids" << std::endl;
    for (size_t x = 0; x < centroids_.rows(); ++x) {
        for (size_t y = 0; y < centroids_.cols(); ++y) {
            std::cout << centroids_(x, y);
            std::cout << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "Centroids (Reference)" << std::endl;
    for (size_t x = 0; x < centroids_.rows(); ++x) {
        for (size_t y = 0; y < centroids_.cols(); ++y) {
            std::cout << reference_centroids_(x, y);
            std::cout << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "Cluster Mass" << std::endl;
    for (auto& m : cluster_mass_) {
        std::cout << m << " ";
    }
    std::cout << std::endl;

    std::cout << "Cluster Mass (Reference)" << std::endl;
    for (auto& m : reference_cluster_mass_) {
        std::cout << m << " ";
    }
    std::cout << std::endl;
}

template class Clustering::ClusteringBenchmark<float, uint32_t, uint32_t, true>;
template class Clustering::ClusteringBenchmark<double, uint64_t, uint64_t, true>;
