/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 * 
 * 
 * Copyright (c) 2016-2018, Lutz, Clemens <lutzcle@cml.li>"
 */

#ifndef KMEANS_INITIALIZER_HPP
#define KMEANS_INITIALIZER_HPP

#include "kmeans_common.hpp"
#include "matrix.hpp"

namespace Clustering {

template <typename PointT>
class KmeansInitializer {
public:
    static void forgy(
            cle::Matrix<PointT, std::allocator<PointT>, size_t, true> const& points,
            cle::Matrix<PointT, std::allocator<PointT>, size_t, true>& centroids
            );

    static void first_x(
            cle::Matrix<PointT, std::allocator<PointT>, size_t, true> const& points,
            cle::Matrix<PointT, std::allocator<PointT>, size_t, true>& centroids
            );
};

using KmeansInitializer32 =
    KmeansInitializer<float>;
using KmeansInitializer64 =
    KmeansInitializer<double>;

}

extern template class Clustering::KmeansInitializer<float>;
extern template class Clustering::KmeansInitializer<double>;

#endif /* KMEANS_INITIALIZER_HPP */
