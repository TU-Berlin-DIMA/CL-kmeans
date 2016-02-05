/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright (c) 2016, Lutz, Clemens <lutzcle@cml.li>
 */

#include "clustering_benchmark.hpp"

#include <cstdint>
#include <algorithm> // std::equal

cle::ClusteringBenchmarkStats::ClusteringBenchmarkStats(const uint32_t num_runs)
    :
        microseconds(num_runs),
        kmeans_stats(num_runs),
        num_runs_(num_runs)
{}

void cle::ClusteringBenchmarkStats::print_times() {
    std::cout << this->num_runs_ << " runs, in µs: [";
    for (uint32_t r = 0; r < microseconds.size(); ++r) {
        std::cout << microseconds[r];
        if (r != microseconds.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;
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
int cle::ClusteringBenchmark<FP, INT, AllocFP, AllocINT, COL_MAJOR>::verify(ClusteringFunction f) {

    cle::KmeansStats stats;
    int is_correct;

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

    is_correct = std::equal(
            reference_labels_.begin(),
            reference_labels_.end(),
            labels_.begin());

    return is_correct;
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
template class cle::ClusteringBenchmark<float, uint32_t, cle::AlignedAllocatorFP32, cle::AlignedAllocatorINT32, true>;
template class cle::ClusteringBenchmark<double, uint64_t, cle::AlignedAllocatorFP64, cle::AlignedAllocatorINT64, true>;
