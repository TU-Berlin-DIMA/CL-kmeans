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
#include <boost/compute/container/mapped_view.hpp>
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

template <typename FP, typename INT, typename AllocFP, typename AllocINT, bool COL_MAJOR>
cle::ClusteringBenchmark<FP, INT, AllocFP, AllocINT, COL_MAJOR>::ClusteringBenchmark(
        const uint32_t num_runs,
        const INT num_points,
        const uint32_t max_iterations,
        cle::Matrix<FP, AllocFP, INT, COL_MAJOR>&& points
        )
    :
        num_runs_(num_runs),
        num_points_(num_points),
        num_clusters_(0),
        max_iterations_(max_iterations),
        points_(std::move(points)),
        labels_(num_points)
{}

template <typename FP, typename INT, typename AllocFP, typename AllocINT, bool COL_MAJOR>
int cle::ClusteringBenchmark<FP, INT, AllocFP, AllocINT, COL_MAJOR>::initialize(
        const INT num_clusters, const INT num_features,
        InitCentroidsFunction init_centroids
        ) {

    num_clusters_ = num_clusters;
    init_centroids_ = init_centroids;

    centroids_.resize(num_clusters, num_features);
    cluster_mass_.resize(num_clusters);

    return 1;
}

template <typename FP, typename INT, typename AllocFP, typename AllocINT, bool COL_MAJOR>
int cle::ClusteringBenchmark<FP, INT, AllocFP, AllocINT, COL_MAJOR>::finalize() {
    return 1;
}

template <typename FP, typename INT, typename AllocFP, typename AllocINT, bool COL_MAJOR>
cle::ClusteringBenchmarkStats cle::ClusteringBenchmark<FP, INT, AllocFP, AllocINT, COL_MAJOR>::run(
        ClusteringFunction f) {

    cle::Timer timer;
    ClusteringBenchmarkStats bs(this->num_runs_);
    bs.set_dimensions(points_.cols(), points_.rows(), centroids_.rows());
    bs.set_types<FP, INT>();

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

template <typename FP, typename INT, typename AllocFP, typename AllocINT, bool COL_MAJOR>
cle::ClusteringBenchmarkStats cle::ClusteringBenchmark<FP, INT, AllocFP, AllocINT, COL_MAJOR>::run(
        ClClusteringFunction f) {

    cle::Timer timer;
    ClusteringBenchmarkStats bs(this->num_runs_);
    bs.set_dimensions(points_.cols(), points_.rows(), centroids_.rows());
    bs.set_types<FP, INT>();

    MappedView<PointT> points_view(
            this->points_.data(),
            this->points_.size());
    VectorPtr<MassT> masses = std::make_shared<Vector<MassT>>(
            this->cluster_mass_);
    VectorPtr<LabelT> labels = std::make_shared<Vector<LabelT>>(
            this->labels_);

    for (uint32_t r = 0; r < this->num_runs_; ++r) {
        init_centroids_(
                points_,
                centroids_
                );

        VectorPtr<PointT> centroids = std::make_shared<Vector<PointT>>(
                this->centroids_.get_data());

        timer.start();
        f(
                max_iterations_,
                points_.cols(),
                points_view,
                centroids,
                masses,
                labels
         );
        bs.microseconds[r] = timer.stop<std::chrono::microseconds>();
    }

    return bs;
}

template <typename FP, typename INT, typename AllocFP, typename AllocINT, bool COL_MAJOR>
void cle::ClusteringBenchmark<FP, INT, AllocFP, AllocINT, COL_MAJOR>::setVerificationReference(std::vector<INT, AllocINT>&& reference_labels) {

    reference_labels_ = std::move(reference_labels);
}

template <typename FP, typename INT, typename AllocFP, typename AllocINT, bool COL_MAJOR>
int cle::ClusteringBenchmark<FP, INT, AllocFP, AllocINT, COL_MAJOR>::setVerificationReference(
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

template<typename FP, typename INT, typename AllocFP, typename AllocINT, bool COL_MAJOR>
uint64_t cle::ClusteringBenchmark<FP, INT, AllocFP, AllocINT, COL_MAJOR>::verify(ClusteringFunction f) {

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
    for (INT l = 0; l < labels_.size(); ++l) {
        if (reference_labels_[l] != labels_[l]) {
            ++counter;
        }
    }

    return counter;
}

template<typename FP, typename INT, typename AllocFP, typename AllocINT, bool COL_MAJOR>
uint64_t cle::ClusteringBenchmark<FP, INT, AllocFP, AllocINT, COL_MAJOR>::verify(ClClusteringFunction f) {

    init_centroids_(
            points_,
            centroids_
            );

    MappedView<PointT> points_view(
            this->points_.data(),
            this->points_.size());
    VectorPtr<PointT> centroids = std::make_shared<Vector<PointT>>(
            this->centroids_.get_data());
    VectorPtr<MassT> masses = std::make_shared<Vector<MassT>>(
            this->cluster_mass_);
    VectorPtr<LabelT> labels = std::make_shared<Vector<LabelT>>(
            this->labels_);

    f(
            max_iterations_,
            points_.cols(),
            points_view,
            centroids,
            masses,
            labels
     );

    boost::compute::copy(
            labels->begin(),
            labels->end(),
            this->labels_.begin());

    uint64_t counter = 0;
    for (INT l = 0; l < labels_.size(); ++l) {
        if (reference_labels_[l] != labels_[l]) {
            ++counter;
        }
    }

    return counter;
}

template<typename FP, typename INT, typename AllocFP, typename AllocINT, bool COL_MAJOR>
void cle::ClusteringBenchmark<FP, INT, AllocFP, AllocINT, COL_MAJOR>::print_labels() {

    std::cout << "Point Label" << std::endl;
    for (INT i = 0; i < labels_.size(); ++i) {
        std::cout << i << " " << labels_[i] << std::endl;
    }
}

template class cle::ClusteringBenchmark<float, uint32_t, std::allocator<float>, std::allocator<uint32_t>, true>;
template class cle::ClusteringBenchmark<double, uint64_t, std::allocator<double>, std::allocator<uint64_t>, true>;
