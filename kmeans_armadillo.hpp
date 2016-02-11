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

#include <armadillo>

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
            KmeansStats&
            );

private:
    arma::Mat<FP> points_;
};

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
}

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

#endif /* KMEANS_ARMADILLO_HPP */
