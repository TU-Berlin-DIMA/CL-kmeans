/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 * 
 * 
 * Copyright (c) 2016, Lutz, Clemens <lutzcle@cml.li>
 */

#include "kmeans_initializer.hpp"

#include <random>

template <typename FP, typename Alloc, typename INT>
void cle::KmeansInitializer<FP, Alloc, INT>::forgy(
        cle::Matrix<FP, Alloc, INT, true> const& points,
        cle::Matrix<FP, Alloc, INT, true>& centroids) {

    std::random_device rand;

    for (INT c = 0; c != centroids.rows(); ++c) {
        INT random_point = rand() % points.rows();
        for (INT d = 0; d < centroids.cols(); ++d) {
            centroids(c, d) = points(random_point, d);
        }
    }
}

template <typename FP, typename Alloc, typename INT>
void cle::KmeansInitializer<FP, Alloc, INT>::first_x(
        cle::Matrix<FP, Alloc, INT, true> const& points,
        cle::Matrix<FP, Alloc, INT, true>& centroids) {

    for (INT d = 0; d < centroids.cols(); ++d) {
        for (INT c = 0; c != centroids.rows(); ++c) {
            centroids(c, d) = points(c % points.rows(), d);
        }
    }
}

template class cle::KmeansInitializer<float, std::allocator<float>, uint32_t>;
template class cle::KmeansInitializer<double, std::allocator<double>, uint64_t>;
template class cle::KmeansInitializer<float, cle::AlignedAllocatorFP32, uint32_t>;
template class cle::KmeansInitializer<double, cle::AlignedAllocatorFP64, uint64_t>;
