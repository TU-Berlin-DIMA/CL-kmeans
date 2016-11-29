/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 * 
 * 
 * Copyright (c) 2016, Lutz, Clemens <lutzcle@cml.li>
 */

#include "kmeans_armadillo.hpp"
#include "timer.hpp"

#include <armadillo>
#include <memory> // std::move
#include <string>

template <typename FP, typename INT, typename AllocFP, typename AllocINT>
char const* cle::KmeansArmadillo<FP, INT, AllocFP, AllocINT>::name() const {

    return "Kmeans_Armadillo";
}

template <typename FP, typename INT, typename AllocFP, typename AllocINT>
int cle::KmeansArmadillo<FP, INT, AllocFP, AllocINT>::initialize(
        cle::Matrix<FP, AllocFP, INT, true> const& points) {

    arma::Mat<FP> new_points(
            points.data(),
            points.rows(),
            points.cols());

    arma::inplace_trans(new_points);

    points_ = std::move(new_points);

    return 1;
}

template <typename FP, typename INT, typename AllocFP, typename AllocINT>
int cle::KmeansArmadillo<FP, INT, AllocFP, AllocINT>::finalize() {
    return 1;
}

template <typename FP, typename INT, typename AllocFP, typename AllocINT>
void cle::KmeansArmadillo<FP, INT, AllocFP, AllocINT>::operator() (
        uint32_t const max_iterations,
        cle::Matrix<FP, AllocFP, INT, true> const&,
        cle::Matrix<FP, AllocFP, INT, true>& centroids,
        std::vector<INT, AllocINT>&,
        std::vector<INT, AllocINT>&,
        Measurement::Measurement& stats) {

    stats
        .add_datapoint(Measurement::DataPointType::PointsBuffer)
        .add_value()
        = points_.n_elem * sizeof(FP);

    stats.add_datapoint(Measurement::DataPointType::CentroidsBuffer)
        .add_value()
        = centroids.size() * sizeof(FP);

    // No data copy, reference existing centroids
    arma::Mat<FP> arma_centroids(
            centroids.data(),
            centroids.rows(),
            centroids.cols(),
            false,
            true);

    arma::inplace_trans(arma_centroids);

    Timer::Timer cpu_timer;
    cpu_timer.start();

    arma::kmeans(
            arma_centroids,
            points_,
            arma_centroids.n_rows,
            arma::keep_existing,
            max_iterations,
            false);

    uint64_t total_time =
        cpu_timer.stop<std::chrono::nanoseconds>();

    stats
        .add_datapoint(Measurement::DataPointType::TotalTime)
        .add_value()
        = total_time;

    stats.set_parameter(
            Measurement::ParameterType::NumIterations,
            std::to_string(max_iterations)
            );

    arma::inplace_trans(arma_centroids);
}

template class cle::KmeansArmadillo<
    float,
    uint32_t,
    std::allocator<float>,
    std::allocator<uint32_t>>;
template class cle::KmeansArmadillo<
    double,
    uint64_t,
    std::allocator<double>,
    std::allocator<uint64_t>>;
#ifdef USE_ALIGNED_ALLOCATOR
template class cle::KmeansArmadillo<
    float,
    uint32_t,
    cle::AlignedAllocatorFP32,
    cle::AlignedAllocatorINT32>;
template class cle::KmeansArmadillo<
    double,
    uint64_t,
    cle::AlignedAllocatorFP64,
    cle::AlignedAllocatorINT64>;
#endif
