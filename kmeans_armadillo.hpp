/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 * 
 * 
 * Copyright (c) 2016, Lutz, Clemens <lutzcle@cml.li>
 */

#ifndef KMEANS_ARMADILLO_HPP
#define KMEANS_ARMADILLO_HPP

#include "kmeans_common.hpp"
#include "matrix.hpp"
#include "measurement/measurement.hpp"

#include <armadillo>
#include <memory>

namespace cle {

template <typename FP, typename INT, typename AllocFP, typename AllocINT>
class KmeansArmadillo {
public:
    char const* name() const;

    int initialize(
            Matrix<FP, AllocFP, INT, true> const& points);
    int finalize();

    void operator() (
            uint32_t const max_iterations,
            cle::Matrix<FP, AllocFP, INT, true> const&,
            cle::Matrix<FP, AllocFP, INT, true>& centroids,
            std::vector<INT, AllocINT>&,
            std::vector<INT, AllocINT>&,
            Measurement::Measurement&
            );

private:
    arma::Mat<FP> points_;
};

using KmeansArmadillo32 = KmeansArmadillo<
    float,
    uint32_t,
    std::allocator<float>,
    std::allocator<uint32_t>>;
using KmeansArmadillo64 = KmeansArmadillo<
    double,
    uint64_t,
    std::allocator<double>,
    std::allocator<uint64_t>>;
#ifdef USE_ALIGNED_ALLOCATOR
using KmeansArmadillo32Aligned = KmeansArmadillo<
    float,
    uint32_t,
    AlignedAllocatorFP32,
    AlignedAllocatorINT32>;
using KmeansArmadillo64Aligned = KmeansArmadillo<
    double,
    uint64_t,
    AlignedAllocatorFP64,
    AlignedAllocatorINT64>;
#endif
}

extern template class cle::KmeansArmadillo<
    float,
    uint32_t,
    std::allocator<float>,
    std::allocator<uint32_t>>;
extern template class cle::KmeansArmadillo<
    double,
    uint64_t,
    std::allocator<double>,
    std::allocator<uint64_t>>;
#ifdef USE_ALIGNED_ALLOCATOR
extern template class cle::KmeansArmadillo<
    float,
    uint32_t,
    cle::AlignedAllocatorFP32,
    cle::AlignedAllocatorINT32>;
extern template class cle::KmeansArmadillo<
    double,
    uint64_t,
    cle::AlignedAllocatorFP64,
    cle::AlignedAllocatorINT64>;
#endif

#endif /* KMEANS_ARMADILLO_HPP */
