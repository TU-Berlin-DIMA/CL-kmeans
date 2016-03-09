/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright (c) 2016, Lutz, Clemens <lutzcle@cml.li>
 */

#include "clustering_benchmark.hpp"

#include <boost/filesystem/path.hpp>

#include <iostream>
#include <fstream>
#include <cstdint>
#include <algorithm> // std::equal
#include <string>
#include <chrono>
#include <vector>
#include <deque>

#include <unistd.h>

#include <Version.h>

char const *const cle::ClusteringBenchmarkStats::parameters_suffix_ =
"_info";

char const *const cle::ClusteringBenchmarkStats::iterated_measurements_suffix_ =
"_iter";

char const *const cle::ClusteringBenchmarkStats::onetime_measurements_suffix_ =
"_once";

uint32_t const cle::ClusteringBenchmarkStats::max_hostname_length_ = 30;
uint32_t const cle::ClusteringBenchmarkStats::max_datetime_length_ = 30;

char const *const cle::ClusteringBenchmarkStats::timestamp_format_ =
"%F-%H-%M-%S";

cle::ClusteringBenchmarkStats::ClusteringBenchmarkStats(const uint32_t num_runs)
    :
        microseconds(num_runs),
        kmeans_stats(num_runs),
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

    boost::filesystem::path file_path(csv_file);
    boost::filesystem::path file_parent = file_path.parent_path();
    boost::filesystem::path file_stem = file_path.stem();
    boost::filesystem::path file_suffix = file_path.extension();

    char hostname[max_hostname_length_];
    gethostname(hostname, max_hostname_length_);

    boost::filesystem::path input_file_path(input_file);
    boost::filesystem::path input_file_name = input_file_path.filename();

    assert(microseconds.size() == kmeans_stats.size());
    for (uint32_t run = 0; run < microseconds.size(); ++run) {

        char datetime[max_datetime_length_];
        std::chrono::system_clock::time_point chrono_date =
            kmeans_stats[run].get_run_date();
        std::time_t timet_date =
            std::chrono::system_clock::to_time_t(chrono_date);
        std::tm *timeinfo_date = std::gmtime(&timet_date);
        std::strftime(
                datetime,
                max_datetime_length_,
                timestamp_format_,
                timeinfo_date
                );

        boost::filesystem::path parameters_file;
        parameters_file += file_parent;
        parameters_file /= datetime;
        parameters_file += '_';
        parameters_file += file_stem;
        parameters_file += parameters_suffix_;
        parameters_file += file_suffix;

        boost::filesystem::path onetime_file;
        onetime_file += file_parent;
        onetime_file /= datetime;
        onetime_file += '_';
        onetime_file += file_stem;
        onetime_file += onetime_measurements_suffix_;
        onetime_file += file_suffix;

        boost::filesystem::path iterated_file;
        iterated_file += file_parent;
        iterated_file /= datetime;
        iterated_file += '_';
        iterated_file += file_stem;
        iterated_file += iterated_measurements_suffix_;
        iterated_file += file_suffix;

        std::ofstream paf(parameters_file.c_str(),
                std::ios_base::out | std::ios::trunc);

        paf << "Timestamp";
        paf << ',';
        paf << "Version";
        paf << ',';
        paf << "Filename";
        paf << ',';
        paf << "Hostname";
        paf << ',';
        paf << "Device";
        paf << ',';
        paf << "NumIters";
        paf << ',';
        paf << "TimeUnit";
        paf << ',';
        paf << "SpaceUnit";
        paf << ',';
        paf << "NumFeatures";
        paf << ',';
        paf << "NumPoints";
        paf << ',';
        paf << "NumClusters";

        paf << '\n';

        paf << datetime;
        paf << ',';
        paf << GIT_REVISION;
        paf << ',';
        paf << input_file_name.c_str();
        paf << ',';
        paf << hostname;
        paf << ',';
        paf << kmeans_stats[run].get_device_name();
        paf << ',';
        paf << kmeans_stats[run].iterations;
        paf << ',';
        paf << "us";
        paf << ',';
        paf << "byte";
        paf << ',';
        paf << num_features_;
        paf << ',';
        paf << num_points_;
        paf << ',';
        paf << num_clusters_;

        paf << '\n';

        paf.close();
        paf.clear();


        std::ofstream otf(onetime_file.c_str(),
                std::ios_base::out | std::ios::trunc);

        otf << "Timestamp";
        otf << ',';
        otf << "TotalTime";

        for (int p = 0; p < cle::DataPoint::get_num_types(); ++p) {
            otf << ',';
            otf << cle::DataPoint::type_to_name((cle::DataPoint::Type) p);
        }

        std::vector<cle::BufferInfo>& bi = kmeans_stats[run].buffer_info;
        for (uint32_t b = 0; b < bi.size(); ++b) {
            otf << ',';
            otf << bi[b].get_name();
        }

        otf << '\n';

        otf << datetime;
        otf << ',';
        otf << microseconds[run];

        std::deque<cle::DataPoint>& dp = kmeans_stats[run].data_points;
        for (int t = 0; t < cle::DataPoint::get_num_types(); ++t) {
            otf << ',';

            for (size_t p = 0; p < dp.size(); ++p) {
                if (dp[p].get_iteration() == -1 && dp[p].get_type() == t) {
                    otf << dp[p].get_nanoseconds() / 1000;
                }
            }
        }

        for (uint32_t b = 0; b < bi.size(); ++b) {
            otf << ',';
            otf << bi[b].get_size();
        }

        otf << '\n';

        otf.close();
        otf.clear();


        std::ofstream itf(iterated_file.c_str(),
                std::ios_base::out | std::ios::trunc);

        itf << "Timestamp";
        itf << ',';
        itf << "Iteration";

        for (int p = 0; p < cle::DataPoint::get_num_types(); ++p) {
            itf << ',';
            itf << cle::DataPoint::type_to_name((cle::DataPoint::Type) p);
        }

        itf << '\n';

        for (int iter = 0; iter < kmeans_stats[run].iterations; ++iter) {
            itf << datetime;
            itf << ',';
            itf << iter;

            for (int t = 0; t < cle::DataPoint::get_num_types(); ++t) {
                itf << ',';

                for (size_t p = 0; p < dp.size(); ++p) {
                    if (dp[p].get_iteration() == iter && dp[p].get_type() == t) {
                        itf << dp[p].get_nanoseconds() / 1000;
                    }
                }
            }

            itf << '\n';
        }

        itf.close();
        itf.clear();
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
                bs.kmeans_stats[r]
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

    cle::KmeansStats stats;

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

    cle::KmeansStats stats;

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
void cle::ClusteringBenchmark<FP, INT, AllocFP, AllocINT, COL_MAJOR>::print_labels() {

    std::cout << "Point Label" << std::endl;
    for (INT i = 0; i < labels_.size(); ++i) {
        std::cout << i << " " << labels_[i] << std::endl;
    }
}

template class cle::ClusteringBenchmark<float, uint32_t, std::allocator<float>, std::allocator<uint32_t>, true>;
template class cle::ClusteringBenchmark<double, uint64_t, std::allocator<double>, std::allocator<uint64_t>, true>;
#ifdef USE_ALIGNED_ALLOCATOR
template class cle::ClusteringBenchmark<float, uint32_t, cle::AlignedAllocatorFP32, cle::AlignedAllocatorINT32, true>;
template class cle::ClusteringBenchmark<double, uint64_t, cle::AlignedAllocatorFP64, cle::AlignedAllocatorINT64, true>;
#endif
