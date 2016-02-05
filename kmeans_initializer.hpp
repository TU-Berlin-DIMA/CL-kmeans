/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 * 
 * 
 * Copyright (c) 2016, Lutz, Clemens <lutzcle@cml.li>
 */

#ifndef KMEANS_INITIALIZER_HPP
#define KMEANS_INITIALIZER_HPP

#include "kmeans_common.hpp"
#include "matrix.hpp"

namespace cle {

template <typename FP, typename Alloc, typename INT>
class KmeansInitializer {
public:
    static void forgy(
            cle::Matrix<FP, Alloc, INT, true> const& points,
            cle::Matrix<FP, Alloc, INT, true>& centroids
            );

    static void first_x(
            cle::Matrix<FP, Alloc, INT, true> const& points,
            cle::Matrix<FP, Alloc, INT, true>& centroids
            );
};

using KmeansInitializer32 =
    KmeansInitializer<float, std::allocator<float>, uint32_t>;
using KmeansInitializer64 =
    KmeansInitializer<double, std::allocator<double>, uint64_t>;
using KmeansInitializer32Aligned =
    KmeansInitializer<float, AlignedAllocatorFP32, uint32_t>;
using KmeansInitializer64Aligned =
    KmeansInitializer<double, AlignedAllocatorFP64, uint64_t>;

}

extern template class cle::KmeansInitializer<float, std::allocator<float>, uint32_t>;
extern template class cle::KmeansInitializer<double, std::allocator<double>, uint64_t>;
extern template class cle::KmeansInitializer<float, cle::AlignedAllocatorFP32, uint32_t>;
extern template class cle::KmeansInitializer<double, cle::AlignedAllocatorFP64, uint64_t>;

#endif /* KMEANS_INITIALIZER_HPP */
