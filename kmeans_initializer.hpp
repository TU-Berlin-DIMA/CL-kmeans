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

#include <vector>

namespace cle {

template <typename FP, typename Alloc>
class KmeansInitializer {
public:
    static void forgy(
            std::vector<FP, Alloc> const& points_x,
            std::vector<FP, Alloc> const& points_y,
            std::vector<FP, Alloc>& centroids_x,
            std::vector<FP, Alloc>& centroids_y);

    static void first_x(
            std::vector<FP, Alloc> const &points_x,
            std::vector<FP, Alloc> const& points_y,
            std::vector<FP, Alloc>& centroids_x,
            std::vector<FP, Alloc>& centroids_y);
};

using KmeansInitializer32 =
    KmeansInitializer<float, std::allocator<float>>;
using KmeansInitializer64 =
    KmeansInitializer<double, std::allocator<double>>;
using KmeansInitializer32Aligned =
    KmeansInitializer<float, AlignedAllocatorFP32>;
using KmeansInitializer64Aligned =
    KmeansInitializer<double, AlignedAllocatorFP64>;

}

extern template class cle::KmeansInitializer<float, std::allocator<float>>;
extern template class cle::KmeansInitializer<double, std::allocator<double>>;
extern template class cle::KmeansInitializer<float, cle::AlignedAllocatorFP32>;
extern template class cle::KmeansInitializer<double, cle::AlignedAllocatorFP64>;

#endif /* KMEANS_INITIALIZER_HPP */
